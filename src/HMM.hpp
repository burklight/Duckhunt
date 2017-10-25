#ifndef _HMM_H_
#define _HMM_H_

#include <iostream>
#include <vector>
#include <cmath>
#include <random>

#define MAXEPOCH 50    // Maximum number of iterations for the Baum-Welch algorithm
#define EPSILON 10e-50  // Really small number to avoid divisions by 0
#define CONVERGENCE 10e-5   /*  Number from which we consider the difference
                                between probabilities negligible */


const double LOGZERO = std::numeric_limits<double>::quiet_NaN();

/* Scaling method for the different algorithms of the HMM */
enum ScMethod
{
    NO_SCALING,     /*  The classical algorithms of alpha and beta pass */
    LOG_SCALING,    /*  Classical algorithms using logarithms.
                        Ref: Numerically Stable Hidden Markov Model Implementation,
                        by Tobias P. Mann */
    CONST_SCALING,  /*  Clever scaling implementation of the classical algorithms
                        using constants.
                        Ref: A Revealing Introduction to Hidden Markov Models,
                        by Mark Stamp */
};

class HMM
/* This class tries to represent a Hidden Markov Model and its different
   functionalities */
{

public:
  /* Class constructors and destructors */
  HMM() {};
  HMM(std::vector<std::vector<double> > _A, std::vector<std::vector<double> > _B,
      std::vector<double> _pi);
  HMM(unsigned int N, unsigned int K);
  HMM(const HMM &model);

  /* Getters */
  unsigned int getNStates() const {return N;}
  unsigned int getNObservations() const {return K;}
  std::vector<std::vector<double> > getA() const {return A;}
  std::vector<std::vector<double> > getB() const {return B;}
  std::vector<double> getPi() const {return pi;}
  std::vector<std::vector<double> > getAlpha() const {return alpha;}
  std::vector<std::vector<double> > getBeta() const {return beta;}
  std::vector<std::vector<double> > getGamma() const {return gamma;}
  std::vector<std::vector<std::vector<double> > > getDiGamm() const {return diGamma;}


  /* functionalities of the HMM */
  std::vector<double> getNextObsProbDist(std::vector<double> currentProbDist,
    unsigned int type);
  std::vector<double> getNextObsProbDist(std::vector<int> sequence,
	  unsigned int type);
  double getObsSeqProb(std::vector<int> sequence, unsigned int type);
  std::vector<int> getStateSequence(std::vector<int> sequence, unsigned int type);
  void estimateModel(std::vector<int> sequence, unsigned int type);
  void estimateModel(std::vector <std::vector<int> > sequences, unsigned int type);
  unsigned int getNextMostProbObs(std::vector<int> sequence, double &maxProb);

private:
  unsigned int N;   // Number of states
  unsigned int K;   // Number of observations
  std::vector<std::vector<double> > A;  // Transition matrix of the HMM
  std::vector<std::vector<double> > B;  // Emission matrix of the HMM
  std::vector<double> pi;  // Initial state distribution of the HMM
  std::vector<std::vector<double> > alpha;  // Alpha value matrix of the HMM
  std::vector<std::vector<double> > beta;   // Beta value matrix of the HMM
  std::vector<std::vector<double> > gamma;  // Gamma value matrix of the HMM
  std::vector<std::vector<std::vector<double> > > diGamma;  // Di-Gamma value matrix of the HMM
  std::vector<double> c;  // Constant value vector for the scaling (Stamp)

  /* Hidden functionalities of the HMM */
  void alphaPass(std::vector<int> sequence, unsigned int type);
  void betaPass(std::vector<int> sequence, unsigned int type);
  void computeGammaAndDiGamma(std::vector<int> sequence, unsigned int type);
};

/* Necessary functions for the logarithm treatment */
double eexp(double x);
double eln(double x);
double elnsum(double x, double y);
double elnproduct(double x, double y);

/* Necessary functions for inference given common denominators */
std::vector<double> normalizeProbDist(std::vector<double> v);

#endif
