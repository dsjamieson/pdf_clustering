#include "density_pdf_clustering.hpp"

DensityPDFClustering::DensityPDFClustering(double t_min_density_ratio, double t_max_density_ratio, int t_num_sample_points, int t_num_mesh_1d, int t_num_threads) {
	start_time = std::chrono::steady_clock::now();
	last_time = std::chrono::steady_clock::now();
	num_threads = t_num_threads;
	num_mesh_1d = t_num_mesh_1d;
	num_mesh_r = num_mesh_1d*num_mesh_1d*num_mesh_1d;
	num_mesh_k = num_mesh_1d*num_mesh_1d*(num_mesh_1d/2 + 1);
	min_density = t_min_density_ratio;
	max_density = t_max_density_ratio;
	log_min_density = log10(min_density);
	log_max_density = log10(max_density);
	num_sample_points = t_num_sample_points;
	max_num_kernel_widths = 4.;
	fprintf(stdout, "Using %d OMP threads\n", num_threads);
}

void DensityPDFClustering::computePDFClustering(char * t_snapshot_filename_base, char * t_volumes_snapshot_filename_base, 
									   char * t_volumes_filename, char * t_pdf_clustering_filename) {

	fprintf(stdout, "Loading Voronoi volumes..."); fflush(stdout);
	readGadgetHeader(t_snapshot_filename_base);
	readVolumes(t_volumes_filename);
	printTimedDone(2);

	fprintf(stdout, "Computing kernel width..."); fflush(stdout);
	computeKDEWidth();
	printTimedDone(2);

	fprintf(stdout, "Distributing Gadget to mesh..."); fflush(stdout);
	loadGadgetCIC(t_snapshot_filename_base);
	printTimedDone(2);

	fprintf(stdout, "Computing matter power spectrum..."); fflush(stdout);
	computeMatterPowerSpectrum();
	writeVector(t_pdf_clustering_filename, wave_numbers);
	appendVector(t_pdf_clustering_filename, matter_powers);
	std::vector<double>	kde_sample_points_output(kde_sample_points.size());
	for(unsigned long i = 0; i < kde_sample_points.size(); i++)
		kde_sample_points_output[i] = pow(10., kde_sample_points[i]);
	appendVector(t_pdf_clustering_filename, kde_sample_points_output);
	printTimedDone(1);

	fprintf(stdout, "Distributing volumes to meshes..."); fflush(stdout);
	loadVoroCIC(t_volumes_snapshot_filename_base);
	appendVector(t_pdf_clustering_filename, mean_density_pdf);
	printTimedDone(1);

	fprintf(stdout, "Computing PDF-matter power spectra...");fflush(stdout);
	computePDFCrossPowerSpectrum(t_pdf_clustering_filename);
	printTimedDone(1);
	
	fprintf(stdout, "Done\n");
	return;
}

void DensityPDFClustering::readGadgetHeader(char * t_snapshot_filename_base) {
	GadgetHeader snapshot_header;
	char snapshot_filename[200];
	sprintf(snapshot_filename, "%s.%d", t_snapshot_filename_base, 0);
	FILE * snapshot_file;
	if(!(snapshot_file = fopen(snapshot_filename, "r"))) {
		if(!(snapshot_file = fopen(t_snapshot_filename_base, "r"))) {
			printf("can't open file `%s` or `%s`\n", snapshot_filename, t_snapshot_filename_base);
			exit(0);
		}
	}
	fseek(snapshot_file, sizeof(int), SEEK_CUR);
	fread(&snapshot_header, sizeof(GadgetHeader), 1, snapshot_file);
	box_length = snapshot_header.box_length*1.e-3;
	num_snapshot_files = snapshot_header.num_files;
	total_num_particles = snapshot_header.total_num_particles[1];
	fclose(snapshot_file);
	fundamental_wave_number = 2.*PI_VALUE::PI/box_length;
	bin_length = box_length/num_mesh_1d; 
	box_volume = pow(box_length, 3);
	bin_volume = pow(bin_length, 3);
	mean_num_bin_particles = (double) total_num_particles/num_mesh_r;
	power_normalization = bin_volume*bin_volume/box_volume;
	return;
}	

void DensityPDFClustering::readVolumes(char * t_volumes_filename) {
	float volumes_file_box_volume;
	unsigned int num_volumes;
	FILE * volumes_file = fopen(t_volumes_filename, "rb");
	if(!volumes_file) {
		fprintf(stdout, "Error, could not open volumes file for reading\n");
		exit(1);
	}
	fseek(volumes_file, sizeof(unsigned int), SEEK_CUR);
	fread(&volumes_file_box_volume, sizeof(float), 1, volumes_file);
	fread(&num_volumes, sizeof(unsigned int), 1, volumes_file);
	volumes.clear();
	volumes.resize(num_volumes);
	fread(volumes.data(), sizeof(float), num_volumes, volumes_file);
	fclose(volumes_file);
	if(volumes_file_box_volume != box_volume) {
		fprintf(stdout, "Error, box volume in volume file does not match Gadget file volume\n");
		exit(0);
	}
	pdf_mean_density = volumes.size()/box_volume;
	return;
}

void DensityPDFClustering::computeKDEWidth() {
	double mean_log_density_ratio = 0.;
	double mean_squared_log_density_ratio = 0.;
	std::vector<float> log_density_ratios(volumes.size()/2);
	#pragma omp parallel
	{
		float log_density_ratio;
		#pragma omp for
		for(unsigned long i = 0; i < volumes.size()/2; i++) {
			log_density_ratio = log10(1./volumes[2*i]/pdf_mean_density);
			log_density_ratios[i] = log_density_ratio;
			log_density_ratio = log10(1./volumes[2*i + 1]/pdf_mean_density);
			log_density_ratios[i] += log_density_ratio;
		}
		#pragma omp barrier
		#pragma omp single
		{
			if(volumes.size()%2 == 1) {
				log_density_ratio = log10(1./volumes[volumes.size()-1]/pdf_mean_density);
				log_density_ratios[0] += log_density_ratio; 
			}
		}
	}
	while(log_density_ratios.size() > 1) {
		if(log_density_ratios.size()%2 == 1) {
			log_density_ratios[0] += log_density_ratios[log_density_ratios.size()-1];
			log_density_ratios.pop_back(); 
		}
		#pragma omp parallel for
		for(unsigned long i = 0; i < log_density_ratios.size()/2; i++) {
			log_density_ratios[i] +=  log_density_ratios[i + log_density_ratios.size()/2];
		}
		log_density_ratios.resize(log_density_ratios.size()/2);
	}
	mean_log_density_ratio = log_density_ratios[0]/volumes.size();
	std::vector<float> squared_log_density_ratios(volumes.size()/2);
	#pragma omp parallel
	{
		float squared_log_density_ratio;
		#pragma omp for
		for(unsigned long i = 0; i < volumes.size()/2; i++) {
			squared_log_density_ratio = log10(1./volumes[2*i]/pdf_mean_density);
			squared_log_density_ratio *= squared_log_density_ratio;
			squared_log_density_ratios[i] = squared_log_density_ratio;
			squared_log_density_ratio = log10(1./volumes[2*i+1]/pdf_mean_density);
			squared_log_density_ratio *= squared_log_density_ratio;
			squared_log_density_ratios[i] += squared_log_density_ratio;
		}
		#pragma omp barrier
		#pragma omp single
		{
			if(volumes.size()%2 == 1) {
				squared_log_density_ratio = log10(1./volumes[volumes.size()-1]/pdf_mean_density);
				squared_log_density_ratio *= squared_log_density_ratio;
				squared_log_density_ratios[0] += squared_log_density_ratio; 
			}
		}
	}
	while(squared_log_density_ratios.size() > 1) {
		if(squared_log_density_ratios.size()%2 == 1) {
			squared_log_density_ratios[0] += squared_log_density_ratios[squared_log_density_ratios.size()-1];
			squared_log_density_ratios.pop_back(); 
		}
		#pragma omp parallel for
		for(unsigned long i = 0; i < squared_log_density_ratios.size()/2; i++) {
			squared_log_density_ratios[i] += squared_log_density_ratios[i + squared_log_density_ratios.size()/2];
		}
		squared_log_density_ratios.resize(squared_log_density_ratios.size()/2);
	}
	mean_squared_log_density_ratio = squared_log_density_ratios[0]/volumes.size();
	double sigma_log_density_ratio = sqrt((mean_squared_log_density_ratio - mean_log_density_ratio*mean_log_density_ratio)*volumes.size()/(volumes.size() - 1.));
	kernel_width = 0.9*sigma_log_density_ratio*pow(volumes.size()/num_mesh_r, -1./5.);
	max_edge_distance = max_num_kernel_widths*kernel_width;
	min_edge_density = log_min_density - max_edge_distance;
	max_edge_density = log_max_density + max_edge_distance;
	sample_width = (log_max_density - log_min_density)/(num_sample_points - 1.);
	num_sample_points_per_sample = (int) round(2.*max_edge_distance/sample_width);
	half_num_sample_points_per_sample = num_sample_points_per_sample/2;
	kde_sample_points.clear();
	kde_sample_points.resize(num_sample_points);
	for(unsigned long i = 0; i < kde_sample_points.size(); i++) {
		kde_sample_points[i] = log_min_density + i*sample_width;
	}
	return;
}

void DensityPDFClustering::loadGadgetCIC(char * t_snapshot_filename_base) {
	density_mesh.clear();
	density_mesh.resize(num_mesh_r);
	#pragma omp parallel for
	for(unsigned long i = 0; i < density_mesh.size(); i++) {
		density_mesh[i] = 0;
	}
	#pragma omp parallel for
	for(int i = 0; i < num_snapshot_files; i++){
		char snapshot_filename[200];
		FILE * snapshot_file;
		GadgetHeader snapshot_header;
		std::vector<float> temp_position(3);
		std::vector<double> position(3);
		int num_particles;	
		if(num_snapshot_files > 1)
			sprintf(snapshot_filename, "%s.%d", t_snapshot_filename_base, i);
		else
			sprintf(snapshot_filename, "%s", t_snapshot_filename_base);
		if(!(snapshot_file = fopen(snapshot_filename, "r"))) {
			printf("can't open file `%s`\n", snapshot_filename);
			exit(0);
		}
		fseek(snapshot_file, sizeof(int), SEEK_CUR);
		fread(&snapshot_header, sizeof(snapshot_header), 1, snapshot_file);
		fseek(snapshot_file, sizeof(int), SEEK_CUR);
		fseek(snapshot_file, sizeof(int), SEEK_CUR);
		num_particles = (int) snapshot_header.num_particles[1];
		for(int n = 0; n < num_particles; n++) {
			fread(temp_position.data(), sizeof(float), 3, snapshot_file);
			for (int j = 0; j < 3; j++){
				position[j] =  ((double) temp_position[j])*1.e-3;
			}
			distributeCIC(position);
		}
		fclose(snapshot_file);
	}
	#pragma omp parallel for
	for(int i = 0; i < num_mesh_r; i++) {
		density_mesh[i] = density_mesh[i]/mean_num_bin_particles - 1. ;
	}
	return;
}

void DensityPDFClustering::distributeCIC(std::vector<double> & t_position) {
	std::vector<int> mesh_indices(6);
	std::vector<double> weights(6);
	int index;
	double weight;
	for(int i = 0; i < 3; i++) {
		mesh_indices[i] = (int) floor(t_position[i]/bin_length);
		mesh_indices[i + 3] = (mesh_indices[i] + 1)%num_mesh_1d;
		weights[i] = 1. + mesh_indices[i] - t_position[i]/bin_length;
		weights[i + 3] = 1. - weights[i];
	}
	for(int i = 0; i < 2; i++) {
		for(int j = 0; j < 2; j++) {
			for(int k = 0; k < 2; k++) {
				index = getMeshIndex(mesh_indices[3*i], mesh_indices[1 + 3*j], mesh_indices[2 + 3*k]);
				weight = weights[2 + 3*k]*weights[1 + 3*j]*weights[3*i];
				#pragma omp atomic
				density_mesh[index] += weight;
			}
		}
	}
	return;
}

void DensityPDFClustering::loadVoroCIC(char * t_volumes_snapshot_filename_base) {
	density_pdf_meshes.clear();
	density_pdf_meshes.resize(num_sample_points);
	density_mesh.clear();
	density_mesh.resize(num_mesh_r);
	mean_density_pdf.clear();
	mean_density_pdf.resize(num_sample_points);
	for(unsigned long i = 0; i < density_pdf_meshes.size(); i++) {
		mean_density_pdf[i] = 0.;
		density_pdf_meshes[i].resize(num_mesh_r);
	}
	#pragma omp parallel for
	for(unsigned long i = 0; i < density_mesh.size(); i++) {
		density_mesh[i] = 0.;
		for(unsigned long j = 0; j < density_pdf_meshes.size(); j++)
			density_pdf_meshes[j][i] = 0.;
	}
	#pragma omp parallel for
	for(int i = 0; i < num_snapshot_files; i++){
		char snapshot_filename[200];
		FILE * snapshot_file;
		FILE * id_file;
		GadgetHeader snapshot_header;
		float temp_position[3];
		std::vector<double> position(3);
		double volume, log_density_ratio;
		unsigned int num_particles, id;
		if(num_snapshot_files > 1)
			sprintf(snapshot_filename, "%s.%d", t_volumes_snapshot_filename_base, i);
		else
			sprintf(snapshot_filename, "%s", t_volumes_snapshot_filename_base);
		if(!(snapshot_file = fopen(snapshot_filename, "r"))) {
			printf("can't open file `%s`\n", snapshot_filename);
			exit(0);
		}
		if(!(id_file = fopen(snapshot_filename, "r"))) {
			printf("can't open file `%s`\n", snapshot_filename);
			exit(0);
		}
		fseek(snapshot_file, sizeof(int), SEEK_CUR);
		fread(&snapshot_header, sizeof(snapshot_header), 1, snapshot_file);
		if(snapshot_header.total_num_particles[1] != volumes.size()) {
			fprintf(stdout, "Error, volumes positions file inconsistent with volumes file, number of tracers does not match\n");
			exit(0);
		}
		if(snapshot_header.box_length*1.e-3 != box_length) {
			fprintf(stdout, "Error, volumes positions file inconsistent with volumes file, box_length does not match\n");
			exit(0);
		}
		fseek(snapshot_file, 2*sizeof(int), SEEK_CUR);
		num_particles = snapshot_header.num_particles[1];
		fseek(id_file, 7*sizeof(int) + sizeof(snapshot_header) + 6*sizeof(float)*num_particles, SEEK_CUR);
		for(unsigned int n = 0; n < num_particles; n++) {
			fread(temp_position, sizeof(float), 3, snapshot_file);
			fread(&id, sizeof(unsigned int), 1, id_file);
			for (int j = 0; j < 3; j++){
				position[j] =  ((double) temp_position[j])*1.e-3;
			}
			volume = (double) volumes[id];
			log_density_ratio = log10(1./volume/pdf_mean_density);
			distributePDFCIC(position, log_density_ratio, volume);
		}
		fclose(snapshot_file);
		fclose(id_file);
	}
	for(int i = 0; i < num_mesh_r; i++) {
		#pragma omp parallel for
		for(unsigned long j = 0; j < density_pdf_meshes.size(); j++) {
			if(density_mesh[i] != 0.) {
				mean_density_pdf[j] += density_pdf_meshes[j][i];
				if(density_mesh[i] != 0)
					density_pdf_meshes[j][i] /= density_mesh[i];
			}
		}
	}
	/*
	char filename[200];
	sprintf(filename, "test_hists.dat");
	FILE * testfile = fopen(filename, "wb");
	std::vector<float> outvector(num_sample_points);
	std::vector<float> xs(kde_sample_points.size());
	for(int i = 0; i < num_sample_points; i++) {
		outvector[i] = (float) pow(10., kde_sample_points[i]);
		xs[i] = outvector[i];
	}
	std::vector<float> dxs(kde_sample_points.size(), 1.);
	*//*
	for(int i = 0; i < num_sample_points-1; i++) {
		dxs[i] = (float) kde_sample_points[i+1] - kde_sample_points[i];
	}
	dxs[num_sample_points-1] = 2*kde_sample_points[num_sample_points-1] - kde_sample_points[num_sample_points-2];
	*//*
	fwrite(&num_sample_points, sizeof(unsigned int), 1, testfile);
	fwrite(outvector.data(), sizeof(float), num_sample_points, testfile);
	for(int i = 0; i < num_sample_points; i++)
		outvector[i] = (float) mean_density_pdf[i]/xs[i]/volumes.size()/log(10.)/dxs[i];
	fwrite(&num_sample_points, sizeof(unsigned int), 1, testfile);
	fwrite(outvector.data(), sizeof(float), num_sample_points, testfile);
	float norm;
	for(int i = 0; i < num_mesh_r; i++) {
		for(int j = 0; j < num_sample_points; j++)
			outvector[j] = (float) density_pdf_meshes[j][i]/xs[j]/log(10.)/dxs[j];
		fwrite(&num_sample_points, sizeof(unsigned int), 1, testfile);
		norm = (float) density_mesh[i];
		fwrite(&norm, sizeof(float), 1, testfile);
		fwrite(outvector.data(), sizeof(float), num_sample_points, testfile);
	}
	fclose(testfile);
	*/
	for(unsigned long i = 0; i < density_pdf_meshes.size(); i++) {
		mean_density_pdf[i] /= volumes.size();
		#pragma omp parallel for
		for(int j = 0;  j < num_mesh_r; j++) {
			if(mean_density_pdf[i] != 0.)
				density_pdf_meshes[i][j] = density_pdf_meshes[i][j]/mean_density_pdf[i] - 1.;	
			else
				density_pdf_meshes[i][j] = 0.;
		}	
		mean_density_pdf[i] = mean_density_pdf[i]/pow(10., kde_sample_points[i])/log(10.);
	}
	return;
}

void DensityPDFClustering::distributePDFCIC(std::vector<double> & t_position, double & t_log_density_ratio, double & t_volume) {
	std::vector<int> mesh_indices(6);
	std::vector<double> weights(6);
	int index;
	double weight;
	for(int i = 0; i < 3; i++) {
		mesh_indices[i] = (int) floor(t_position[i]/bin_length);
		mesh_indices[i + 3] = (mesh_indices[i] + 1)%num_mesh_1d;
		weights[i + 3] = t_position[i]/bin_length - mesh_indices[i];
		weights[i] = 1. - weights[i + 3];
	}
	for(int i = 0; i < 2; i++) {
		for(int j = 0; j < 2; j++) {
			for(int k = 0; k < 2; k++) {
				index = getMeshIndex(mesh_indices[3*i], mesh_indices[1 + 3*j], mesh_indices[2 + 3*k]);
				weight = weights[3*i]*weights[1 + 3*j]*weights[2 + 3*k];
				distributeSampleKernel(index, t_log_density_ratio, weight);
			}
		}
	}
	return;
}

void DensityPDFClustering::distributeSampleKernel(int & t_index, double & t_log_density_ratio, double & t_weight) {
	if(min_edge_density < t_log_density_ratio && max_edge_density > t_log_density_ratio) {
		int start_sample_index = (int) floor((t_log_density_ratio - log_min_density)/sample_width) - half_num_sample_points_per_sample;
		if(start_sample_index < 0)
			start_sample_index = 0;
		int max_sample_index = start_sample_index + num_sample_points_per_sample;
		if(max_sample_index >= num_sample_points)
			max_sample_index = num_sample_points - 1;
		for(int i = start_sample_index; i <= max_sample_index; i++) {
			#pragma omp atomic
			density_pdf_meshes[i][t_index] += getGaussianKernalValue(kde_sample_points[i], kernel_width, t_log_density_ratio)*t_weight;
		}
	}
	#pragma omp atomic
	density_mesh[t_index] += t_weight;
	return;
}

double DensityPDFClustering::getGaussianKernalValue(double & t_mean, double & t_sigma, double & t_x) {
	return exp(-pow((t_x - t_mean)/t_sigma , 2)/2.)/sqrt(2.*PI_VALUE::PI)/t_sigma;
}

void DensityPDFClustering::computeMatterPowerSpectrum() {
	fftw_init_threads(); 
	fftw_plan_with_nthreads(num_threads);  
	density_modes = (fftw_complex*) fftw_malloc(num_mesh_k * sizeof(fftw_complex));
	fftw_plan density_plan = fftw_plan_dft_r2c_3d(num_mesh_1d, num_mesh_1d, num_mesh_1d, density_mesh.data(), density_modes, FFTW_ESTIMATE);
	fftw_execute(density_plan);
	fftw_destroy_plan(density_plan);
	wave_numbers.clear();
	num_modes.clear();
	matter_powers.clear();
	wave_numbers.resize(num_mesh_1d/2, 0.);
	num_modes.resize(num_mesh_1d/2, 0.);
	matter_powers.resize(num_mesh_1d/2, 0.);
	#pragma omp parallel
	{
		std::vector<double> thread_wave_numbers(num_mesh_1d/2, 0.);
		std::vector<double> thread_num_modes(num_mesh_1d/2, 0.);
		std::vector<double> thread_powers(num_mesh_1d/2, 0.);	
		int i_k, j_k, k_k, bin_index, mode_index;
		double weight, wave_number, power;
		#pragma omp for nowait
		for(int i = 0; i < num_mesh_1d; i++) {
			for(int j = 0; j < num_mesh_1d; j++) {
				for(int k = 0; k < num_mesh_1d/2 + 1; k++) {
					if(k == 0 || (num_mesh_1d%2 == 0 && k == num_mesh_1d/2)) {
						if(j > num_mesh_1d/2)
							continue;
						if(j == 0 || (num_mesh_1d%2 == 0 && j == num_mesh_1d/2)) {
							if(i > num_mesh_1d/2 || i ==0)
								continue;
						}
					}
					i_k = i;
					if (i_k > num_mesh_1d/2) i_k -= num_mesh_1d;
					j_k = j;
					if (j_k > num_mesh_1d/2) j_k -= num_mesh_1d;
					k_k = k;
					if (k_k > num_mesh_1d/2) k_k -= num_mesh_1d;
					wave_number = fundamental_wave_number*sqrt(i_k*i_k + j_k*j_k + k_k*k_k);
					weight = pow(sinc(i_k*PI_VALUE::PI/num_mesh_1d)*sinc(j_k*PI_VALUE::PI/num_mesh_1d)*sinc(k_k*PI_VALUE::PI/num_mesh_1d), 4);
					bin_index = ((int) round(wave_number/fundamental_wave_number)) - 1;
					if(bin_index > (int) wave_numbers.size())
						continue;
					mode_index = getModeIndex(i, j, k);
					power = (density_modes[mode_index][0]*density_modes[mode_index][0] + density_modes[mode_index][1]*density_modes[mode_index][1])/weight;
					thread_wave_numbers[bin_index] += wave_number;
					thread_powers[bin_index] += power;
					thread_num_modes[bin_index]++;
				}
			}
		}
		#pragma omp critical
		{
			for(unsigned long i = 0; i < wave_numbers.size(); i++) {
				wave_numbers[i] += thread_wave_numbers[i];
				matter_powers[i] += thread_powers[i];
				num_modes[i] += thread_num_modes[i];
			}
		}
	}
	fftw_cleanup_threads();
	#pragma omp parallel for
 	for(unsigned long i = 0; i < wave_numbers.size(); i++){
		if(num_modes[i] > 0) {
  			wave_numbers[i] /= num_modes[i];
			matter_powers[i] *= power_normalization/num_modes[i];
		}
  	}
	return;
}

void DensityPDFClustering::computePDFCrossPowerSpectrum(char * t_powers_filename) {
	fftw_init_threads(); 
	fftw_plan_with_nthreads(num_threads); 
	fftw_complex * density_pdf_modes = (fftw_complex*) fftw_malloc(num_mesh_k * sizeof(fftw_complex));
	fftw_plan density_pdf_plan;
	pdf_powers.clear();
	pdf_powers.resize(num_mesh_1d/2);
	cross_powers.clear();
	cross_powers.resize(num_mesh_1d/2);
	for(unsigned long ind = 0; ind < density_pdf_meshes.size(); ind++) {
		#pragma omp parallel for
		for(int i = 0; i < num_mesh_1d/2; i++) {
			pdf_powers[i] = 0.;
			cross_powers[i] = 0.;
		}
		density_pdf_plan = fftw_plan_dft_r2c_3d(num_mesh_1d, num_mesh_1d, num_mesh_1d, density_pdf_meshes[ind].data(), density_pdf_modes, FFTW_ESTIMATE);
		fftw_execute(density_pdf_plan);
		#pragma omp parallel
		{
			std::vector<double> thread_pdf_powers(num_mesh_1d/2, 0.);
			std::vector<double> thread_cross_powers(num_mesh_1d/2, 0.);
			std::vector<double> thread_num_modes(num_mesh_1d/2, 0.);
			int i_k, j_k, k_k, bin_index, mode_index;
			double weight, wave_number; 
			#pragma omp for 
			for(int i = 0; i < num_mesh_1d; i++) {
				for(int j = 0; j < num_mesh_1d; j++) {
					for(int k = 0; k < num_mesh_1d/2 + 1; k++) {
						if(k == 0 || (num_mesh_1d%2 == 0 && k == num_mesh_1d/2)) {
							if(j > num_mesh_1d/2)
								continue;
							if(j == 0 || (num_mesh_1d%2 == 0 && j == num_mesh_1d/2)) {
								if(i > num_mesh_1d/2 || i == 0)
									continue;
							}
						}
						i_k = i;
						if (i_k > num_mesh_1d/2) i_k -= num_mesh_1d;
						j_k = j;
						if (j_k > num_mesh_1d/2) j_k -= num_mesh_1d;
						k_k = k;
						if (k_k > num_mesh_1d/2) k_k -= num_mesh_1d;
						wave_number = fundamental_wave_number*sqrt(i_k*i_k + j_k*j_k + k_k*k_k);
						weight = pow(sinc(i_k*PI_VALUE::PI/num_mesh_1d)*sinc(j_k*PI_VALUE::PI/num_mesh_1d)*sinc(k_k*PI_VALUE::PI/num_mesh_1d), 4);
						bin_index = ((int) round(wave_number/fundamental_wave_number)) - 1;
						if(bin_index > num_mesh_1d/2 - 1)
							continue;
						mode_index = getModeIndex(i, j, k);
						thread_pdf_powers[bin_index] += (density_pdf_modes[mode_index][0]*density_pdf_modes[mode_index][0] + density_pdf_modes[mode_index][1]*density_pdf_modes[mode_index][1])/weight;
						thread_cross_powers[bin_index] += (density_modes[mode_index][0]*density_pdf_modes[mode_index][0] + density_modes[mode_index][1]*density_pdf_modes[mode_index][1])/weight;
					}
				}
			}
			#pragma omp critical
			{
				for(int i = 0; i < num_mesh_1d/2; i++) {
 					pdf_powers[i] += thread_pdf_powers[i];
					cross_powers[i] += thread_cross_powers[i];
				}
			}
		}
		#pragma omp parallel for
	 	for(unsigned long i = 0; i < wave_numbers.size(); i++){
			if(num_modes[i] > 0) {
				pdf_powers[i] *= power_normalization/num_modes[i];
				cross_powers[i] *= power_normalization/num_modes[i];
			}
  		}
		appendVector(t_powers_filename, cross_powers);
		appendVector(t_powers_filename, pdf_powers);
	}
	return;
}

void DensityPDFClustering::writeVector(char * t_filename, std::vector<float> & t_vector) {
	FILE * file = fopen(t_filename, "wb");
	if (!file){
		fprintf(stdout, "Cannot open file for power spectra output\n");
		exit(0);
	}
	unsigned int size = (unsigned int) t_vector.size();
	fwrite(&size, sizeof(unsigned int), 1, file);
	fwrite(t_vector.data(), sizeof(float), size, file);
	fclose(file);
	return;
}

void DensityPDFClustering::appendVector(char * t_filename, std::vector<float> & t_vector) {
	FILE * file = fopen(t_filename, "ab");
	if (!file){
		fprintf(stdout, "Cannot open file for power spectra output\n");
		exit(0);
	}
	unsigned int size = (unsigned int) t_vector.size();
	fwrite(&size, sizeof(unsigned int), 1, file);
	fwrite(t_vector.data(), sizeof(float), size, file);
	fclose(file);
	return;
}

void DensityPDFClustering::writeVector(char * t_filename, std::vector<double> & t_vector) {
	FILE * file = fopen(t_filename, "wb");
	if (!file){
		fprintf(stdout, "Cannot open file for power spectra output\n");
		exit(0);
	}
	unsigned int size = (unsigned int) t_vector.size();
	fwrite(&size, sizeof(unsigned int), 1, file);
	std::vector<float> vector(t_vector.begin(), t_vector.end());
	fwrite(vector.data(), sizeof(float), size, file);
	fclose(file);
	return;
}

void DensityPDFClustering::appendVector(char * t_filename, std::vector<double> & t_vector) {
	FILE * file = fopen(t_filename, "ab");
	if (!file){
		fprintf(stdout, "Cannot open file for power spectra output\n");
		exit(0);
	}
	unsigned int size = (unsigned int) t_vector.size();
	fwrite(&size, sizeof(unsigned int), 1, file);
	std::vector<float> vector(t_vector.begin(), t_vector.end());
	fwrite(vector.data(), sizeof(float), size, file);
	fclose(file);
	return;
}

double DensityPDFClustering::sinc(double x){
	if (x==0) return 1;
	return sin(x)/x;
}

int DensityPDFClustering::getMeshIndex(int t_i, int t_j, int t_k) {
	return t_k + num_mesh_1d*(t_j + num_mesh_1d*t_i);
}

int DensityPDFClustering::getModeIndex(int t_i, int t_j, int t_k) {
	return t_k + (num_mesh_1d/2 + 1)*(t_j + num_mesh_1d*t_i);
}

void DensityPDFClustering::printTimedDone(int t_num_tabs) {
	std::chrono::steady_clock::time_point this_time = std::chrono::steady_clock::now();
	float step_time = (float) std::chrono::duration_cast<std::chrono::milliseconds>(this_time - last_time).count()*1.e-3;
	float cumulative_time = (float) std::chrono::duration_cast<std::chrono::milliseconds>(this_time - start_time).count()*1.e-3;
	int step_time_h = (int) floor(step_time/60./60.);
	int step_time_m = (int) floor(step_time/60. - step_time_h*60);
	float step_time_s = step_time - step_time_m*60. - step_time_h*60.*60.;
	int cumulative_time_h = (int) floor(cumulative_time/60./60.);
	int cumulative_time_m = (int) floor(cumulative_time/60. - cumulative_time_h*60);
	float cumulative_time_s = cumulative_time - cumulative_time_m*60. - cumulative_time_h*60.*60.;
	for(int i =0; i < t_num_tabs; i++)
		fprintf(stdout, "\t");
	fprintf(stdout, "done ");
	fprintf(stdout, "[%02d:%02d:%05.2f, %02d:%02d:%05.2f]\n", step_time_h, step_time_m, step_time_s, cumulative_time_h, cumulative_time_m, cumulative_time_s);
	last_time = this_time;
	return;
}

DensityPDFClustering::~DensityPDFClustering() {
	fftw_free(density_modes);
}

