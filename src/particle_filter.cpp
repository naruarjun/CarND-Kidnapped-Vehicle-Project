/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 10;
  std::default_random_engine gen;
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);

  for(int i=0;i<num_particles;i++){
    double current_particle_x = dist_x(gen);
    double current_particle_y = dist_y(gen);
    double current_particle_theta = dist_theta(gen);
    Particle current_particle;
    current_particle.id = i;
    current_particle.x = current_particle_x;
    current_particle.y = current_particle_y;
    current_particle.theta = current_particle_theta;
    current_particle.weight = 1.0;
    particles.push_back(current_particle);
    weights.push_back(1.0);
  }
  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  std::default_random_engine gen;
  for(int i=0;i<num_particles;i++){
    Particle current_particle = particles[i];
    double current_x = current_particle.x;
    double current_y = current_particle.y;
    double current_theta = current_particle.theta;
    double final_x, final_y, final_theta;
    if(yaw_rate == 0.0){
      final_x = current_x + (velocity*(cos(current_theta)))*delta_t;
      final_y = current_y + (velocity*(sin(current_theta)))*delta_t;
      final_theta = current_theta + yaw_rate*delta_t;
    }
    else{
      final_x = current_x + (velocity*(sin(current_theta + yaw_rate*delta_t) - sin(current_theta)))/yaw_rate;
      final_y = current_y + (velocity*(cos(current_theta) - cos(current_theta + yaw_rate*delta_t)))/yaw_rate;
      final_theta = current_theta + yaw_rate*delta_t;
    }
    std::normal_distribution<double> dist_x(final_x, std_pos[0]);
    std::normal_distribution<double> dist_y(final_y, std_pos[1]);
    std::normal_distribution<double> dist_theta(final_theta, std_pos[2]);
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen); 
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  for(int i = 0; i<observations.size();i++){
    int closest_landmark = -1;
    double observed_x = observations[i].x;
    double observed_y = observations[i].y;
    double closest_distance = -1;    
    for(int j=0; j<predicted.size();j++){
      int current_id = predicted[j].id;
      double predicted_x = predicted[j].x;
      double predicted_y = predicted[j].y;
      double distance = dist(observed_x, observed_y, predicted_x, predicted_y);
      if(j==0){
        closest_distance = distance;
      }
      if(distance <= closest_distance){
        closest_distance = distance;
        closest_landmark = current_id;
      }
    }
    observations[i].id = closest_landmark;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  double sig_x = std_landmark[0];
  double sig_y = std_landmark[1];
  double total_weight = 0;
  for(int i=0; i<num_particles; i++){
    double mu_x, mu_y, x_obs, y_obs;
    double current_particle_x = particles[i].x;
    double current_particle_y = particles[i].y;
    double current_particle_theta = particles[i].theta;
     
     //Convert Observations from Vehicle coordinate frame to map coordinate frame
    vector<LandmarkObs> transformed_observations;
    for(int j=0;j<observations.size();j++){
      LandmarkObs current;
      current.id = observations[j].id;
      current.x = current_particle_x + (cos(current_particle_theta)*observations[j].x) - (sin(current_particle_theta)*observations[j].y);
      current.y = current_particle_y + (sin(current_particle_theta)*observations[j].x) + (cos(current_particle_theta)*observations[j].y);
      transformed_observations.push_back(current);
    }

    vector<LandmarkObs> map_landmarks_in_range;
    int landmark_id = 0;
    for(int j=0; j<map_landmarks.landmark_list.size(); j++){
      double current_distance = dist(map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f, current_particle_x, current_particle_y);
      if(current_distance<=sensor_range* sqrt(2)){
        LandmarkObs current_landmark;
        current_landmark.id = landmark_id;
        landmark_id += 1;
        current_landmark.x = map_landmarks.landmark_list[j].x_f;
        current_landmark.y = map_landmarks.landmark_list[j].y_f;
        map_landmarks_in_range.push_back(current_landmark);
      }
    }

    dataAssociation(map_landmarks_in_range, transformed_observations);
    
    particles[i].weight = 1.0;
    weights[i] = 1.0;
    double gauss_norm = 1 / (2 * M_PI * sig_x * sig_y);
    for(int j=0; j<transformed_observations.size(); j++){
      if(map_landmarks_in_range.size()==0){
        break;
      }
      double current_map_x = map_landmarks_in_range[transformed_observations[j].id].x;
      double current_map_y = map_landmarks_in_range[transformed_observations[j].id].y;

      double current_x = transformed_observations[j].x;
      double current_y = transformed_observations[j].y;
      double exponent = (pow(current_x - current_map_x, 2) / (2 * pow(sig_x, 2)))
               + (pow(current_y - current_map_y, 2) / (2 * pow(sig_y, 2)));
      
      double current_weight = gauss_norm * exp(-exponent);
       
      particles[i].weight *= current_weight; 
      weights[i] *= current_weight;
    }
    total_weight += particles[i].weight;
    
  }
  for(int i=0;i<num_particles; i++){
    particles[i].weight /= total_weight;
    weights[i] /= total_weight;
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  std:: default_random_engine gen;
  std::discrete_distribution<int> particle_sampler(weights.begin(), weights.end());
  vector<Particle> temp;
  for(int i=0;i<num_particles;i++){
    int index = particle_sampler(gen);
    temp.push_back(particles[index]);
  }
  for(int i=0;i<num_particles;i++){
    particles[i] = temp[i];
  }
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}