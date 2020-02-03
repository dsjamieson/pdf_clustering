#include "periodic_box.hpp"

PeriodicBox::PeriodicBox(char * t_snapshot_filename_base, char * t_snapshot_file_type, int t_num_mesh_1d, int t_num_threads) {
	start_time = std::chrono::steady_clock::now();
	last_time = std::chrono::steady_clock::now();
	num_threads = t_num_threads;
	snapshot_filename_base = t_snapshot_filename_base;
	num_mesh_1d = t_num_mesh_1d;
	num_mesh_r = num_mesh_1d*num_mesh_1d*num_mesh_1d;
	density_mesh.resize(num_mesh_r);
	num_mesh_k = num_mesh_1d*num_mesh_1d*(num_mesh_1d/2 + 1);
	readGadgetHeader(snapshot_filename_base);
	fundamental_wave_number = 2.*PI_VALUE::PI/box_length;
	bin_length = box_length/num_mesh_1d; 
	box_volume = pow(box_length, 3);
	bin_volume = pow(bin_length, 3);
	mean_num_bin_particles = (double) total_num_particles/num_mesh_r;
	power_normalization = bin_volume*bin_volume/box_volume;
	fprintf(stdout, "Using %d OMP threads\n", num_threads);

}

void PeriodicBox::readGadgetHeader(char * t_snapshot_filename_base) {
	gadget_header snapshot_header;
	char snapshot_filename[200];
	sprintf(snapshot_filename, "%s.%d", snapshot_filename_base, 0);
	FILE * snapshot_file;
	if(!(snapshot_file = fopen(snapshot_filename, "r"))) {
		if(!(snapshot_file = fopen(snapshot_filename_base, "r"))) {
			printf("can't open file `%s` or `%s`\n", snapshot_filename, snapshot_filename_base);
			exit(0);
		}
	}
	fseek(snapshot_file, sizeof(int), SEEK_CUR);
	fread(&snapshot_header, sizeof(gadget_header), 1, snapshot_file);
	box_length = snapshot_header.box_length*1.e-3;
	num_snapshot_files = snapshot_header.num_files;
	total_num_particles = snapshot_header.total_num_particles[1];
	fclose(snapshot_file);
	return;
}	

void PeriodicBox::computePDFClustering(char * t_volume_filename, double t_min_density_ratio, double t_max_density_ratio, int t_num_sample_points, char * t_pdf_clustering_filename) {

	min_density = t_min_density_ratio;
	max_density = t_max_density_ratio;
	log_min_density = log(min_density);
	log_max_density = log(max_density);
	num_sample_points = t_num_sample_points;
	max_num_kernel_widths = 4.;

	fprintf(stdout, "Loading Voronoi volumes..."); fflush(stdout);
	readVolumes(t_volume_filename);
	printTimedDone(3);

	fprintf(stdout, "Reading Gadget, distributing to meshes..."); fflush(stdout);
	loadVoroGadgetCIC();
	printTimedDone(1);

	fprintf(stdout, "Computing matter power spectrum..."); fflush(stdout);
	computeMatterPowerSpectrum();
	writePowers(t_pdf_clustering_filename);
	appendPowers(t_pdf_clustering_filename, kde_sample_points);
	printTimedDone(2);

	fprintf(stdout, "Computing PDF-matter power spectra...");fflush(stdout);
	computePDFCrossPowerSpectrum(t_pdf_clustering_filename);
	printTimedDone(2);
	
	fprintf(stdout, "Done\n");
	return;
}

void PeriodicBox::readVolumes(char * t_volume_filename) {
	double mean_density = 0.;
	double mean_log_density_ratio = 0.;
	double mean_log_density_ratio_squared = 0.;
	double sigma_log_density_ratio = 0.;
	double volume_sum = 0.;
	#pragma omp parallel 
	{
		int num_threads = omp_get_num_threads();
		int thread_id = omp_get_thread_num();
		unsigned int num_volume_samples;
		double thread_mean_density = 0.;
		double thread_mean_log_density_ratio = 0.;
		double thread_mean_log_density_ratio_squared = 0.;
		double thread_volume_sum = 0.;
		FILE * volume_file = fopen(t_volume_filename, "rb");
		if(!volume_file) {
			#pragma omp single
			fprintf(stdout, "Error, could not open volume file for reading\n");
			exit(1);
		}
		fread(&num_volume_samples, sizeof(unsigned int), 1, volume_file);
		#pragma omp single
		{
			log_density_ratios.clear();
			log_density_ratios.resize(num_volume_samples);
			pdf_mean_density = num_volume_samples/box_volume;
		}
		#pragma omp barrier
		unsigned int num_tasks, start_task;
		if(thread_id < (int) num_volume_samples % num_threads) {
			num_tasks = (unsigned int) num_volume_samples/num_threads + 1;
			start_task = (unsigned int) num_tasks*thread_id;
		}
		else {
			num_tasks = (unsigned int) num_volume_samples/num_threads;
			start_task = (unsigned int) num_tasks*thread_id + num_volume_samples % num_threads;		
		}
		fseek(volume_file, start_task*sizeof(float), SEEK_CUR);
		float volume;
		double density;
		for(unsigned int i = start_task; i < num_tasks + start_task; i++) {
			fread(&volume, sizeof(float), 1, volume_file);
			if(volume <= 0) {
				#pragma omp critical
				{
					fprintf(stdout, "\nError: invalid volume encountered %.6e\ncontinuing...", volume);
				}
				volume = 1./pdf_mean_density;
			}
			thread_volume_sum += (double) volume;
			density = (double) 1./volume;
			log_density_ratios[i] = log(density/pdf_mean_density);
			thread_mean_density += density;
			thread_mean_log_density_ratio += log_density_ratios[i];
			thread_mean_log_density_ratio_squared += log_density_ratios[i]*log_density_ratios[i];
		}
		fclose(volume_file);
		#pragma omp critical
		{
			volume_sum += thread_volume_sum;
			mean_density += thread_mean_density/num_volume_samples;
			mean_log_density_ratio += thread_mean_log_density_ratio/((double) num_volume_samples);
			mean_log_density_ratio_squared += thread_mean_log_density_ratio_squared/((double) num_volume_samples);
		}
	}
	sigma_log_density_ratio = sqrt((mean_log_density_ratio_squared - mean_log_density_ratio*mean_log_density_ratio)*(log_density_ratios.size())/log_density_ratios.size() - 1.);
	kernel_width = 0.9*sigma_log_density_ratio*pow(log_density_ratios.size(), -1./5.);
	max_edge_distance = max_num_kernel_widths*kernel_width;
	min_edge_density = log_min_density - max_edge_distance;
	max_edge_density = log_max_density + max_edge_distance;
	sample_width = (log_max_density - log_min_density)/(num_sample_points - 1.);
	num_sample_points_per_sample = (int) round(2.*max_edge_distance/sample_width);
	half_num_sample_points_per_sample = num_sample_points_per_sample/2;
	kde_sample_points.clear();
	kde_sample_points.resize(num_sample_points);
	for(unsigned long i = 0; i < kde_sample_points.size(); i++)
		kde_sample_points[i] = log(min_density) + i*sample_width;
	return;
}

void PeriodicBox::loadVoroGadgetCIC() {
	density_pdf_meshes.clear();
	density_pdf_meshes.resize(num_sample_points);
	density_mesh.clear();
	density_mesh.resize(num_mesh_r);
	for(unsigned long i = 0; i < density_pdf_meshes.size(); i++)
		density_pdf_meshes[i].resize(num_mesh_r);
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
		gadget_header snapshot_header;
		float temp_position[3];
		std::vector<double> position(3);
		unsigned int num_particles, id;
		if(num_snapshot_files > 1)
			sprintf(snapshot_filename, "%s.%d", snapshot_filename_base, i);
		else
			sprintf(snapshot_filename, "%s", snapshot_filename_base);
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
		fseek(snapshot_file, 2*sizeof(int), SEEK_CUR);
		num_particles = snapshot_header.num_particles[1];
		fseek(id_file, 7*sizeof(int) + sizeof(snapshot_header) + 6*sizeof(float)*num_particles, SEEK_CUR);
		for(unsigned int n = 0; n < num_particles; n++) {
			fread(temp_position, sizeof(float), 3, snapshot_file);
			fread(&id, sizeof(unsigned int), 1, id_file);
			for (int j = 0; j < 3; j++){
				position[j] =  ((double) temp_position[j])*1.e-3;
			}
			distributePDFCIC(position, log_density_ratios[id], id);
		}
		fclose(snapshot_file);
		fclose(id_file);
	}
	std::vector<double>	mean_density_pdf(num_sample_points, 0.);
	#pragma omp parallel for
	for(int i = 0; i < num_mesh_r; i++) {
		for(unsigned long j = 0; j < density_pdf_meshes.size(); j++) {
			density_pdf_meshes[j][i] /= density_mesh[i];
			#pragma omp atomic
			mean_density_pdf[j] += density_pdf_meshes[j][i];
		}
		density_mesh[i] = density_mesh[i]/mean_num_bin_particles - 1.;
	}
	for(unsigned long i = 0; i < density_pdf_meshes.size(); i++) {
		mean_density_pdf[i] /= num_mesh_r;
		#pragma omp parallel for
		for(int j = 0;  j < num_mesh_r; j++)
			density_pdf_meshes[i][j] = density_pdf_meshes[i][j]/mean_density_pdf[i] - 1.;	
	}
	return;
}

void PeriodicBox::distributePDFCIC(std::vector<double> & t_position, double & t_log_density_ratio, unsigned int t_id) {
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
				distributeSampleKernel(index, t_log_density_ratio, weight);
			}
		}
	}
	return;
}

void PeriodicBox::distributeSampleKernel(int & t_index, double & t_log_density_ratio, double & weight) {
	if(min_edge_density < t_log_density_ratio && max_edge_density > t_log_density_ratio) {
		int start_sample_index = (int) floor((t_log_density_ratio - log_min_density)/sample_width) - half_num_sample_points_per_sample;
		if(start_sample_index < 0)
			start_sample_index = 0;
		int max_sample_index = start_sample_index + num_sample_points_per_sample;
		if(max_sample_index >= num_sample_points)
			max_sample_index = num_sample_points - 1;
		for(int i = start_sample_index; i <= max_sample_index; i++) {
			#pragma omp atomic
			density_pdf_meshes[i][t_index] += getGaussianKernalValue(kde_sample_points[i], kernel_width, t_log_density_ratio)*weight;
		}
	}
	return;
}

double PeriodicBox::getGaussianKernalValue(double & t_mean, double & t_sigma, double & t_x) {
	return exp(-pow((t_x - t_mean)/t_sigma , 2)/2.)/sqrt(2.*PI_VALUE::PI)/t_sigma;
}
void PeriodicBox::computeMatterPowerSpectrum() {
	fftw_init_threads(); 
	fftw_plan_with_nthreads(num_threads); 
	fftw_complex * density_modes = (fftw_complex*) fftw_malloc(num_mesh_k * sizeof(fftw_complex));
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
	fftw_free(density_modes);
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


void PeriodicBox::computePDFCrossPowerSpectrum(char * t_powers_filename) {
	fftw_init_threads(); 
	fftw_plan_with_nthreads(num_threads); 
	fftw_complex * density_modes = (fftw_complex*) fftw_malloc(num_mesh_k * sizeof(fftw_complex));
	fftw_plan density_plan = fftw_plan_dft_r2c_3d(num_mesh_1d, num_mesh_1d, num_mesh_1d, density_mesh.data(), density_modes, FFTW_ESTIMATE);
	fftw_execute(density_plan);
	fftw_destroy_plan(density_plan);
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
		appendPowers(t_powers_filename, cross_powers);
		appendPowers(t_powers_filename, pdf_powers);
	}
	return;
}

void PeriodicBox::writePowers(char * t_powers_filename) {
	FILE * powers_file = fopen(t_powers_filename, "wb");
	if (!powers_file){
		fprintf(stdout, "Cannot open file for power spectra output\n");
		exit(0);
	}
	unsigned long size = wave_numbers.size();
	fwrite(&size, sizeof(unsigned long), 1, powers_file);
	fwrite(wave_numbers.data(), sizeof(double), size, powers_file);
	fwrite(&size, sizeof(unsigned long), 1, powers_file);
	fwrite(matter_powers.data(), sizeof(double), size, powers_file);
	fclose(powers_file);
	return;
}

void PeriodicBox::appendPowers(char * t_powers_filename, std::vector<double> & t_powers) {
	FILE * powers_file = fopen(t_powers_filename, "ab");
	if (!powers_file){
		fprintf(stdout, "Cannot open file for power spectra output\n");
		exit(0);
	}
	unsigned long size = t_powers.size();
	fwrite(&size, sizeof(unsigned long), 1, powers_file);
	fwrite(t_powers.data(), sizeof(double), size, powers_file);
	fclose(powers_file);
	return;
}

double PeriodicBox::sinc(double x){
	if (x==0) return 1;
	return sin(x)/x;
}

int PeriodicBox::getMeshIndex(int t_i, int t_j, int t_k) {
	return t_k + num_mesh_1d*(t_j + num_mesh_1d*t_i);
}

int PeriodicBox::getModeIndex(int t_i, int t_j, int t_k) {
	return t_k + (num_mesh_1d/2 + 1)*(t_j + num_mesh_1d*t_i);
}

void PeriodicBox::printTimedDone(int t_num_tabs) {
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


