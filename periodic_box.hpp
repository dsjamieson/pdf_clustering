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

struct gadget_header{
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

class PeriodicBox {

	public:
	PeriodicBox(char * t_snapshot_filename_base, char * t_snapshot_filename_type, int t_num_mesh_1d, int t_num_threads);
	PeriodicBox(char * t_modes_filename_base, int t_num_threads);

	void computePDFClustering(char * t_volume_filename, double t_min_density_ratio, double t_max_denisty_ratio, 
							  int t_num_sample_points, char * t_pdf_clustering_filename);
	void computePowerSpectrum();
	void computePowerSpectrumNotPeriodic();
	void computeCrossPowerSpectrum(char * t_tracer_filename, std::vector<double> & tracer_thresholds);
	void computeCrossPowerSpectrum(char * t_modes_filename, char * t_tracer_filename, std::vector<double> & t_tracer_thresholds);
	void writePowerSpectrum(char * t_power_spectrum_filename);
	void writeCrossPowerSpectrum(char * t_power_spectrum_filename, std::vector<double> & t_tracer_thresholds);
	void writeModes(char * t_modes_filename);

	private:
	char * snapshot_filename_base;
	double box_length;
	double box_volume;
	double bin_length;
	double bin_volume;
	double mean_num_bin_particles;
	double fundamental_wave_number;
	double power_normalization;
	int num_threads;
	unsigned int total_num_particles;
	int num_snapshot_files;
	int num_mesh_1d;
	int num_mesh_r; 
	int num_mesh_k;

	double min_density;
	double log_min_density;
	double max_density;
	double log_max_density;
	int num_sample_points;
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
	std::vector<double> log_density_ratios;
	std::vector<double> kde_sample_points;
	std::vector<double> wave_numbers;
	std::vector<double> num_modes;
	std::vector<double> powers;
	std::vector<double> matter_powers;
	std::vector<double> pdf_powers;
	std::vector<double> cross_powers;
	std::vector<double> num_tracers;
	std::chrono::steady_clock::time_point start_time;
	std::chrono::steady_clock::time_point last_time;

	void readGadgetHeader(char * t_snapshot_filename_base);
	void readVolumes(char * t_volume_filename);
	void loadVoroGadgetCIC();
	void distributePDFCIC(double t_position[3], double t_log_density_ratio);
	void distributeSampleKernel(int & t_index, double & t_log_density_ratio, double & weight);
	double getGaussianKernalValue(double & t_mean, double & t_sigma, double & t_x);
	void computeMatterPowerSpectrum();
	void computePDFCrossPowerSpectrum(char * t_powers_filename);
	void writePowers(char * t_powers_filename);
	void appendPowers(char * t_powers_filename, std::vector<double> & t_powers);

	double sinc(double x);
	int getMeshIndex(int t_i, int t_j, int t_k);
	int getModeIndex(int t_i, int t_j, int t_k);
	void printTimedDone(int t_num_tabs);

};


