#include "density_pdf_clustering.hpp"

int main(int argc, char *argv[]){
	if(argc != 10) {
		fprintf(stdout, "Usage: pdf_clustering snapshot_file_base volumes_filename volumes_snapshot_filename_base min_density max_denisty num_sample_points num_mesh_sites_per_dimension pdf_clustering_filename num_threads\n");
		exit(0);
	}
	char * snapshot_filename_base = argv[1]; 
	char * volumes_filename = argv[2];
	char * volumes_snapshot_filename_base = argv[3];
	double min_density;
	try {
		min_density = std::stod(argv[4]);
	}
	catch (...) {
		fprintf(stderr, "Could not convert min_density to double\n");
		exit(0);
	}
	double max_density;
	try {
		max_density = std::stod(argv[5]);
	}
	catch (...) {
		fprintf(stderr, "Could not convert max_density to double\n");
		exit(0);
	}
	int num_sample_points;
	try {
		num_sample_points = std::stoi(argv[6]);
	}
	catch (...) {
		fprintf(stderr, "Could not convert num_sample_points to int\n");
		exit(0);
	}
	int num_mesh_1d; 
	try {
		num_mesh_1d = std::stoi(argv[7]);
	}
	catch (...) {
		fprintf(stderr, "Could not convert num_mesh_1d to int\n");
		exit(0);
	}
	char * pdf_clustering_filename = argv[8];
	int num_threads;
	try {
		num_threads = std::stoi(argv[9]);
	}
	catch (...) {
		fprintf(stderr, "Could not convert num_threads to int\n");
		exit(0);
	}
	DensityPDFClustering density_pdf_clustering(min_density, max_density, num_sample_points , num_mesh_1d, num_threads);
	density_pdf_clustering.computePDFClustering(snapshot_filename_base, volumes_filename, volumes_snapshot_filename_base, pdf_clustering_filename);
	return 0;
}
