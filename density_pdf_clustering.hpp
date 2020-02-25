#include <cstring>
#include <cmath>
#include <fstream>
#include <sstream>
#include <omp.h>
#include <fftw3.h>
#include <vector>
#include <chrono>

namespace PI_VALUE {
    const double PI = 3.14159265358979;
}

struct GadgetHeader{
	unsigned int num_particles[6];
	double masses[6];
	double scale_factor;
	double redshift;
	int flag_sfr;
	int flag_feedback;
	unsigned int total_num_particles[6];
	int flag_cooling;
	int num_files;
	double box_length;
	double omega_m;
	double omega_lambda;
	double hubble;
	char fill[256 - 6*4 - 6*8 - 2*8 - 2*4 - 6*4 - 2*4 - 4*8];	/* fills to 256 Bytes */
};

class DensityPDFClustering {
	public:
	DensityPDFClustering(double t_min_density_ratio, double t_max_density_ratio, int t_num_sample_points, int t_num_mesh_1d, int t_num_threads);
	~DensityPDFClustering();
	void computePDFClustering(char * t_snapshot_filename_base, char * t_volumes_snapshot_filename_base, 
							  char * t_volumes_filename, char * t_pdf_clustering_filename);
	private:
	std::chrono::steady_clock::time_point start_time;
	std::chrono::steady_clock::time_point last_time;
	int num_threads;
	double min_density;
	double log_min_density;
	double max_density;
	double log_max_density;
	int num_sample_points;
	int num_mesh_1d;
	int num_mesh_r; 
	int num_mesh_k;

	double box_length;
	double box_volume;
	double bin_length;
	double bin_volume;
	unsigned int total_num_particles;
	double mean_num_bin_particles;
	double fundamental_wave_number;
	double power_normalization;
	int num_snapshot_files;
	fftw_complex * density_modes;

	double pdf_mean_density;
	double kernel_width;
	double max_num_kernel_widths;
	double max_edge_distance;
	double min_edge_density;
	double max_edge_density;
	double sample_width;
	int num_sample_points_per_sample;
	int half_num_sample_points_per_sample;

	std::vector<double> density_mesh;
	std::vector<std::vector<double>> density_pdf_meshes;
	std::vector<float> volumes;
	std::vector<double> kde_sample_points;
	std::vector<double> wave_numbers;
	std::vector<double> num_modes;
	std::vector<double> matter_powers;
	std::vector<double> pdf_powers;
	std::vector<double> cross_powers;
	std::vector<double>	mean_density_pdf;

	void readGadgetHeader(char * t_snapshot_filename_base);
	void readVolumes(char * t_volumes_filename);
	void computeKDEWidth();
	void loadGadgetCIC(char * t_snapshot_filename_base);
	void distributeCIC(std::vector<double> & t_position);
	void loadVoroCIC(char * t_volumes_snapshot_filename_base);
	void distributePDFCIC(std::vector<double> & t_position, double & t_log_density_ratio, double & t_volume);
	void distributeSampleKernel(int & t_index, double & t_log_density_ratio, double & t_weight);
	double getGaussianKernalValue(double & t_mean, double & t_sigma, double & t_x);
	void computeMatterPowerSpectrum();
	void computePDFCrossPowerSpectrum(char * t_powers_filename);
	void writeVector(char * t_filename, std::vector<double> & t_vector);
	void appendVector(char * t_filename, std::vector<double> & t_vector);
	void writeVector(char * t_filename, std::vector<float> & t_vector);
	void appendVector(char * t_filename, std::vector<float> & t_vector);
	double sinc(double x);
	int getMeshIndex(int t_i, int t_j, int t_k);
	int getModeIndex(int t_i, int t_j, int t_k);
	void printTimedDone(int t_num_tabs);

};


