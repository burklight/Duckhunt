#include "HMM.hpp"

/* Class constructors and destructors */
HMM::HMM(std::vector<std::vector<double> > _A, std::vector<std::vector<double> > _B,
    std::vector<double> _pi)
{
  A = _A;
  B = _B;
  pi = _pi;
  N = _A.size();
  K = _B[0].size();
}

HMM::HMM(unsigned int nStates, unsigned int nObservations)
{
  N = nStates;
  K = nObservations;

  A.resize(N);
  for(unsigned int n = 0; n < N; n++) A[n].resize(N);
  B.resize(N);
  for(unsigned int n = 0; n < N; n++) B[n].resize(K);
  pi.resize(N);

  // For creating similar random numbers to represent the probabilities
  std::random_device random;
  std::mt19937 generator(rand());
  std::normal_distribution<double> gauss(10,1);

  // For making sure that each row adds to 1.0
  double rowSum = 0.0;
  double normalizer = 0.0;

  // Random initialization of pi
  for(unsigned int i = 0; i < N; i++)
  {
    pi[i] = gauss(generator);
    rowSum += pi[i];
  }
  normalizer = rowSum;
  rowSum = 0.0;
  for(unsigned int i = 0; i < N - 1; i++)
  {
    pi[i] /= (normalizer + EPSILON);
    rowSum += pi[i];
  }
  pi[N - 1] = 1.0 - rowSum;

  // Random initialization of A
  for(unsigned int i = 0; i < N; i++)
  {
    rowSum = 0.0;
    for(unsigned int j = 0; j < N; j++)
    {
      A[i][j] = gauss(generator);
      rowSum += A[i][j];
    }
    normalizer = rowSum;
    rowSum = 0.0;
    for(unsigned int j = 0; j < N - 1; j++)
    {
      A[i][j] /= (normalizer + EPSILON);
      rowSum += A[i][j];
    }
    A[i][N - 1] = 1.0 - rowSum;
  }

  // Random initialization of B
  for(unsigned int i = 0; i < N; i++)
  {
    rowSum = 0.0;
    for(unsigned int k = 0; k < K; k++)
    {
      B[i][k] = gauss(generator);
      rowSum += B[i][K];
    }
    normalizer = rowSum;
    rowSum = 0.0;
    for(unsigned int k = 0; k < K - 1; k++)
    {
      B[i][k] /= (normalizer + EPSILON);
      rowSum += B[i][k];
    }
    B[i][K - 1] = 1.0 - rowSum;
  }
}

HMM::HMM(const HMM &model)
{
  A = model.getA();
  B = model.getB();
  pi = model.getPi();
  N = model.getNStates();
  K = model.getNObservations();
}

/* functionalities of the HMM */

std::vector<double> HMM::getNextObsProbDist(std::vector<double> currentProbDist,
  unsigned int type = CONST_SCALING)
/* Computes the probability distribution of the next observation based on the
   current probability distribution */
{
  std::vector<double> nexObsProbDist(K);
  double sum = 0.0;

  switch (type)
  {
    case NO_SCALING:
    case CONST_SCALING:
      for(unsigned int k = 0; k < K; k++)
      {
        nexObsProbDist[k] = 0.0;
        for(unsigned int i = 0; i < N; i++)
        {
          sum = 0.0;
          for(unsigned int j = 0; j < N; j++)
            sum += A[i][j] * B[j][k];
          nexObsProbDist[k] += currentProbDist[i] * sum;
        }
      }
      break;
    case LOG_SCALING:
    /*  Assumes that is receiving the logarithm of a probability distribution and
        returns the logarithm of the next probability distribution */
      for(unsigned int k = 0; k < K; k++)
      {
        nexObsProbDist[k] = LOGZERO;
        for(unsigned int i = 0; i < N; i++)
        {
          sum = LOGZERO;
          for(unsigned int j = 0; j < N; j++)
            sum = elnsum(sum, elnproduct(eln(A[i][j]), eln(B[j][k])));
          nexObsProbDist[k] = elnsum(nexObsProbDist[k],
            elnproduct(currentProbDist[i], sum));
        }
      }
      break;
  }

  return nexObsProbDist;
}

double HMM::getObsSeqProb(std::vector<int> sequence, unsigned int type = CONST_SCALING)
/* Returns the probability of a sequence of observations */
{
  unsigned int T = sequence.size();
  double probability;
  alphaPass(sequence, type);

  switch (type)
  {
    case NO_SCALING:
      probability = 0.0;
      for(unsigned int i = 0; i < N; i++)
        probability += alpha[T - 1][i];
      break;
    case LOG_SCALING:
    /* Assumes that alpha is logarithmic and returns the logarithm of
      the probability */
      probability = LOGZERO;
      for(unsigned int i = 0; i < N; i++)
        probability = elnsum(probability, alpha[T - 1][i]);
      break;
    case CONST_SCALING:
    /* Assumes that alpha is logarithmic and calculated with constant scaling
      and returns the logarithm of the probability */
      unsigned int T = sequence.size();
      for(unsigned int t = 0; t < T; t++)
        probability -= eln(c[t]);
      break;
  }
  return probability;
}

std::vector<int> HMM::getStateSequence(std::vector<int> sequence,
  unsigned int type = CONST_SCALING)
/* Returns the most probable state sequence given the available observations.
   Only the Viterbi decoding algorithm has been implemented due to the imprecision
   of the classical method involving gamma */
{
  unsigned int T = sequence.size();
  std::vector<int> stateIndices(T);

  std::vector<std::vector<double> > delta(T , std::vector<double>(N));
  std::vector<std::vector<unsigned int> > deltaIndices(T , std::vector<unsigned int>(N));

  unsigned int maxIndex = 0;
  double max = 0.0;
  double tmp = 0.0;

  switch (type) {
    case NO_SCALING:
    case CONST_SCALING:
      // Initialize delta
      for(unsigned int i = 0; i < N; i++)
        delta[0][i] = pi[i] * B[i][sequence[0]];

      // For each time step calculate delta and the index of previous max delta
      for(unsigned int t = 1; t < T; t++)
        for(unsigned int i = 0; i < N; i++)
        {
          max = delta[t - 1][0] * A[0][i] * B[i][sequence[t]];
          maxIndex = 0;
          for(unsigned int j = 1; j < N; j++)
          {
            tmp = delta[t - 1][j] * A[j][i] * B[i][sequence[t]];
            if(tmp > max)
            {
              max = tmp;
              maxIndex = j;
            }
          }
          delta[t][i] = max;
          deltaIndices[t][i] = maxIndex;
      }

      maxIndex = 0;
      max = delta[T - 1][0];
      for(unsigned int i = 0; i < N; i++)
        if(delta[T - 1][i] > max)
        {
          maxIndex = i;
          max = delta[T - 1][i];
        }

      // The most probable path is the one determined by the last maximum delta
      //  and its predecessors
      stateIndices[T - 1] = maxIndex;
      for(unsigned int t = T - 1; t > 0; t--)
        stateIndices[t - 1] = deltaIndices[t][stateIndices[t]];
      break;
    case LOG_SCALING:
    /* Ref: A Tutorial on Hidden Markov Models and Selected Applications in
       Speech Regognition
       by Lawrence R. Rabiner */
      // Initialize delta
      for(unsigned int i = 0; i < N; i++)
        delta[0][i] = elnproduct(eln(pi[i]), eln(B[i][sequence[0]]));

      // For each time step calculate delta and the index of previous max delta
      for(unsigned int t = 1; t < T; t++)
        for(unsigned int i = 0; i < N; i++)
        {
          max = elnproduct(delta[t - 1][0], elnproduct(eln(A[0][i]),
            eln(B[i][sequence[t]])));
          maxIndex = 0;
          for(unsigned int j = 1; j < N; j++)
          {
            tmp = elnproduct(delta[t - 1][j], elnproduct(eln(A[j][i]),
              eln(B[i][sequence[t]])));
            if(std::isnan(tmp)) {}
            else if(std::isnan(max) || tmp > max)
            {
              max = tmp;
              maxIndex = j;
            }
          }
          delta[t][i] = max;
          deltaIndices[t][i] = maxIndex;
      }

      maxIndex = 0;
      max = delta[T - 1][0];
      for(unsigned int i = 0; i < N; i++)
        if(std::isnan(delta[T - 1][i])) {}
        else if(std::isnan(max) || delta[T - 1][i] > max)
        {
          maxIndex = i;
          max = delta[T - 1][i];
        }

      // The most probable path is the one determined by the last maximum delta
      //  and its predecessors
      stateIndices[T - 1] = maxIndex;
      for(unsigned int t = T - 1; t > 0; t--)
        stateIndices[t - 1] = deltaIndices[t][stateIndices[t]];
      break;
  }

  return stateIndices;
}

void HMM::estimateModel(std::vector<int> sequence, unsigned int type = CONST_SCALING)
/* Returns the estimation of the HMM given the observations */
{
  unsigned int T = sequence.size();
  unsigned int epoch = 0;
  double prevSequenceProb = -10e-50;
  double num = 0.0;
  double den = 0.0;

  // First alculation of alpha, sequence probability, beta, gamma and di-gamma
  computeGammaAndDiGamma(sequence, type);
  double sequenceProb = getObsSeqProb(sequence);

  do
  {
    if(type == NO_SCALING || type == CONST_SCALING)
    /* If the scaling is a constant the gamma and di-gamma remain the same,
       so there is no need for differenciation */
    {
      // Estimation of pi
      for(unsigned int i = 0; i < N; i++)
        pi[i] = gamma[0][i];

      // Estimation of A
      for(unsigned int i = 0; i < N; i++)
        for(unsigned int j = 0; j < N; j++)
        {
          num = 0.0;
          den = 0.0;
          for(unsigned int t = 0; t < T - 1; t++)
          {
            num += diGamma[t][i][j];
            den += gamma[t][i];
          }
          A[i][j] = num / (den + EPSILON);
        }

      // Estimation of B
      for(unsigned int i = 0; i < N; i++)
        for(unsigned int k = 0; k < K; k++)
        {
          num = 0.0;
          den = 0.0;
          for(unsigned int t = 0; t < T; t++)
          {
            if(sequence[t] == k)
              num += gamma[t][i];
            den += gamma[t][i];
          }
          B[i][k] = num / (den + EPSILON);
        }
    }
    else if(type == LOG_SCALING)
    /* If alpha, beta, gamma and di-gamma are logarithmic, a different estimation
       of pi, A and B must be done */
    {
      // Estimation of pi
      for(unsigned int i = 0; i < N; i++)
        pi[i] = eexp(gamma[0][i]);

      // Estimation of A
      for(unsigned int i = 0; i < N; i++)
        for(unsigned int j = 0; j < N; j++)
        {
          num = LOGZERO;
          den = LOGZERO;
          for(unsigned int t = 0; t < T - 1; t++)
          {
            num = elnsum(num,diGamma[t][i][j]);
            den = elnsum(den,gamma[t][i]);
          }
          A[i][j] = eexp(elnproduct(num, (-1.0 * den)));
        }

      // Estimation of B
      for(unsigned int i = 0; i < N; i++)
        for(unsigned int k = 0; k < K; k++)
        {
          num = LOGZERO;
          den = LOGZERO;
          for(unsigned int t = 0; t < T; t++)
          {
            if(sequence[t] == k)
              num = elnsum(num, gamma[t][i]);
            den = elnsum(den,gamma[t][i]);
          }
          B[i][k] = eexp(elnproduct(num, (-1.0 * den)));
        }
    }

    prevSequenceProb = sequenceProb;
    computeGammaAndDiGamma(sequence, type);
    sequenceProb = getObsSeqProb(sequence);

    epoch++;
  } while((epoch <  MAXEPOCH) && (sequenceProb > prevSequenceProb ||
      std::isnan(prevSequenceProb) || !std::isnan(sequenceProb)));
}

void HMM::estimateModel(std::vector <std::vector<int> > sequences, unsigned int type)
// Re-estimation of parameters given multiple sequences
// Ref.: A tutorial on Hidden Markov model and selected applications in
//       speech recognition.
//       by Lawrence R. Rabiner
{
	unsigned int S = sequences.size(); // number of sequences
	unsigned int T = sequences[0].size();
	unsigned int epoch = 0;
	double num = 0.0;
	double den = 0.0;
	double sequenceWeight = 1.0 / S;
	double seqProb = 1;
	double prevSeqProb = 1;

	//initializing pi to [1, 0, 0 ... 0]
	pi[0] = 1;
	for (int i = 1; i < pi.size(); i++)
		pi[i] = 0;

	do
	{
		std::vector < std::vector<double>> numA(N, std::vector<double>(N, 0.0));
		std::vector<double> denA(N, 0.0);
		std::vector < std::vector<double>> numB(N, std::vector<double>(K, 0.0));
		std::vector<double> denB(N, 0.0);

		for (unsigned int s = 0; s < S; s++)
		{
			if (epoch == 0)
			{
				// First calculation of alpha, sequence probability, beta, gamma and di-gamma
				computeGammaAndDiGamma(sequences[s], type);
				prevSeqProb *= getObsSeqProb(sequences[s]);
			}

			if (type == NO_SCALING || type == CONST_SCALING)
				/* If the scaling is a constant the gamma and di-gamma remain the same,
				so there is no need for differenciation */
			{

				// Estimation of A
				for (unsigned int i = 0; i < N; i++)
					for (unsigned int j = 0; j < N; j++)
					{
						for (unsigned int t = 0; t < sequences[s].size() - 1; t++)
						{
							numA[i][j] += sequenceWeight * diGamma[t][i][j];
							//den[i] is equal for all j and needs to be added only once
							if(j == 0)
								denA[i] += sequenceWeight * gamma[t][i];
						}

					}

				// Estimation of B
				for (unsigned int i = 0; i < N; i++)
					for (unsigned int k = 0; k < K; k++)
					{
						for (unsigned int t = 0; t < sequences[s].size() - 1; t++)
						{
							if (sequences[s][t] == k)
								numB[i][k] += sequenceWeight * gamma[t][i];
							if(k == 0)
								denB[i] += sequenceWeight * gamma[t][i];
						}
					}
			}
		}


		for (unsigned int i = 0; i < N; i++)
			for (unsigned int j = 0; j < N; j++)
				A[i][j] = numA[i][j] / (denA[i] + EPSILON);
		for (unsigned int i = 0; i < N; i++)
			for (unsigned int k = 0; k < K; k++)
				B[i][k] = numB[i][k] / (denB[i] + EPSILON);


		prevSeqProb = seqProb;
		seqProb = 1.0;
		for (unsigned int s = 0; s < S; s++)
		{
				// First calculation of alpha, sequence probability, beta, gamma and di-gamma
				computeGammaAndDiGamma(sequences[s], type);
				seqProb *= getObsSeqProb(sequences[s]);
		}
		epoch++;

	} while ((epoch < MAXEPOCH) && (seqProb > prevSeqProb ||
		std::isnan(prevSeqProb) || !std::isnan(seqProb)));

}

unsigned int HMM::getNextMostProbObs(std::vector<int> sequence, double &maxProb)
/* Calculates the most probable observation in the next time step along with
   its probability given the sequence of observations until now.
   It only works with a logarithmic scaling to prevent underflow problems. */
{
  std::vector<double> nextProbs(K);
  unsigned int T = sequence.size();

  computeGammaAndDiGamma(sequence, CONST_SCALING);

  for(unsigned int k = 0; k < K; k++)
  {
    nextProbs[k] = 0.0;
    for(unsigned int i = 0; i < N; i++)
      for(unsigned int j = 0; j < N; j++)
        nextProbs[k] += gamma[T - 1][i] * A[i][j] * B[j][k];
  }

  nextProbs = normalizeProbDist(nextProbs);

  maxProb = nextProbs[0];
  int maxIndex = 0;
  for(unsigned int k = 1; k < K; k++)
    if(nextProbs[k] > maxProb)
    {
      maxProb = nextProbs[k];
      maxIndex = k;
    }

  return maxIndex;
}

std::vector<double> HMM::getNextObsProbDist(std::vector<int> sequence,
	unsigned int type = CONST_SCALING)
	/* Calculates the probability distribution of the next observation given the
	observations so far.
	Currently only with constant scaling. */
{
	std::vector<double> nextObsProbs(K);
	unsigned int T = sequence.size();

	computeGammaAndDiGamma(sequence, CONST_SCALING);

	for (unsigned int k = 0; k < K; k++)
	{
		nextObsProbs[k] = 0.0;
		for (unsigned int i = 0; i < N; i++)
			for (unsigned int j = 0; j < N; j++)
				nextObsProbs[k] += gamma.back()[i] * A[i][j] * B[j][k];
	}

	return nextObsProbs;
}

/* hidden functionalities of the HMM */
void HMM::alphaPass(std::vector<int> sequence, unsigned int type = CONST_SCALING)
/* Compute the forward algorithm or alpha pass */
{
  unsigned int T = sequence.size();
  alpha.resize(T);
  for(unsigned int t = 0; t < T; t++) alpha[t].resize(N);

  switch (type)
  {
    case NO_SCALING:
      // Initialize alpha
      for(unsigned int i = 0; i < N; i++)
        alpha[0][i] = pi[i] * B[i][sequence[0]];

      // Calculate alpha for each time step
      for(unsigned int t = 1; t < T; t++)
        for(unsigned int i = 0; i < N; i++)
        {
          alpha[t][i] = 0;
          for(unsigned int j = 0; j < N; j++)
            alpha[t][i] += alpha[t - 1][j] * A[j][i];
          alpha[t][i] *= B[i][sequence[t]];
        }
      break;
    case LOG_SCALING:
    /* Computes logarithm of alpha */
      // Initialize alpha
      for(unsigned int i = 0; i < N; i++)
        alpha[0][i] = elnproduct(eln(pi[i]), eln(B[i][sequence[0]]));

      // Calculate alpha for each time step
      for(unsigned int t = 1; t < T; t++)
        for(unsigned int i = 0; i < N; i++)
        {
          alpha[t][i] = LOGZERO;
          for(unsigned int j = 0; j < N; j++)
            alpha[t][i] = elnsum(alpha[t][i], elnproduct(alpha[t - 1][j], eln(A[j][i])));
          alpha[t][i] = elnproduct(alpha[t][i], eln(B[i][sequence[t]]));
        }
      break;
    case CONST_SCALING:
    /* Computes constant scaled alpha using Stamp algorithm */
      c.resize(T);

      // Calculate the initial alpha
      c[0] = 0.0;
      for(unsigned int i = 0; i < N; i++)
      {
        alpha[0][i] = pi[i] * B[i][sequence[0]];
        c[0] += alpha[0][i];
      }

      // Scale the initial alpha
      c[0] = 1 / (c[0] + EPSILON);
      for(unsigned int i = 0; i < N; i++)
        alpha[0][i] = c[0] * alpha[0][i];

      // Calculate alpha for each time step
      for(unsigned int t = 1; t < T; t++)
      {
        c[t] = 0.0;
        for(unsigned int i = 0; i < N; i++)
        {
          alpha[t][i] = 0.0;
          for(unsigned int j = 0; j < N; j++)
            alpha[t][i] += A[j][i] * alpha[t - 1][j];
          alpha[t][i] *= B[i][sequence[t]];
          c[t] += alpha[t][i];
        }

        // Scale alpha for each time step
        c[t] = 1 / (c[t] + EPSILON);
        for(unsigned int i = 0; i < N; i++)
          alpha[t][i] *= c[t];
      }
      break;
  }
}

void HMM::betaPass(std::vector<int> sequence, unsigned int type = CONST_SCALING)
/* Compute the backward algorithm or beta pass */
{
  int T = sequence.size();

  beta.resize(T);
  for(unsigned int t = 0; t < T; t++) beta[t].resize(N);

  switch (type)
  {
    case NO_SCALING:
      // Set the initial (last) beta  to 1
      for(unsigned int i = 0; i < N; i++)
        beta[T - 1][i] = 1.0;

      // Calculate beta for each time step
      for(int t = T - 2; t >= 0; t--)
        for(unsigned int i = 0; i < N; i++)
        {
          beta[t][i] = 0.0;
          for(unsigned int j = 0; j < N; j++)
            beta[t][i] += A[i][j] * B[j][sequence[t + 1]] * beta[t + 1][j];
        }
      break;
    case LOG_SCALING:
    /* Computes logarithm of beta */
      // Set the initial (last) beta  to log(1) = 0
      for(unsigned int i = 0; i < N; i++)
        beta[T - 1][i] = 0.0;

      // Calculate beta for each time step
      for(int t = T - 2; t >= 0; t--)
        for(unsigned int i = 0; i < N; i++)
        {
          beta[t][i] = LOGZERO;
          for(unsigned int j = 0; j < N; j++)
            beta[t][i] = elnsum(beta[t][i], elnproduct(eln(A[i][j]),
              elnproduct(eln(B[j][sequence[t + 1]]), beta[t + 1][j])));
        }
      break;
    case CONST_SCALING:
    /* Computes constant scaled beta using Stamp algorithm */
      // Set the initial (last) beta  with the scaling of last c
      for(unsigned int i = 0; i < N; i++)
        beta[T - 1][i] = c[T - 1];

      // Calculate beta for each time step
      for(int t = T - 2; t >= 0; t--)
        for(unsigned int i = 0; i < N; i++)
        {
          beta[t][i] = 0;
          for(unsigned int j = 0; j < N; j++)
            beta[t][i] += A[i][j] * B[j][sequence[t + 1]] * beta[t + 1][j];

          // Scale beta for each time step (with the same scaling factor than alpha)
          beta[t][i] *= c[t];
        }
      break;
  }
}

void HMM::computeGammaAndDiGamma(std::vector<int> sequence,
  unsigned int type = CONST_SCALING)
/* Compute the gamma and di-gamma matrices */
{
  unsigned int T = sequence.size();
  double normalizer = 0.0;

  alphaPass(sequence, type);  // Compute alpha
  betaPass(sequence, type);   // Compute beta

  diGamma.resize(T - 1);
  for(unsigned int t = 0; t < T - 1; t++)
  {
    diGamma[t].resize(N);
    for(unsigned int i = 0; i < N; i++) diGamma[t][i].resize(N);
  }
  gamma.resize(T);
  for(unsigned int t = 0; t < T; t++) gamma[t].resize(N);

  if(type == NO_SCALING || type == CONST_SCALING)
  /* If the scaling is a constant the divisions of the gamma and di-gamma formulas
     remain the same, so there is no need for differenciation */
  {
    for(unsigned int t = 0; t < T - 1; t++)
    {
      normalizer = 0.0;
      for(unsigned int i = 0; i < N; i++)
        for(unsigned int j = 0; j < N; j++)
        {
          diGamma[t][i][j] = alpha[t][i] * A[i][j] * B[j][sequence[t + 1]]
            * beta[t + 1][j];
          normalizer += diGamma[t][i][j];
        }

      for(unsigned int i = 0; i < N; i++)
      {
        gamma[t][i] = 0.0;
        for(unsigned int j = 0; j < N; j++)
        {
          diGamma[t][i][j] /= (normalizer + EPSILON);
          gamma[t][i] += diGamma[t][i][j];
        }
      }
    }

  /* Special computation for last time step (gamma) */
  normalizer = 0.0;
  for(unsigned int i = 0; i < N; i++)
    normalizer += alpha[T - 1][i];
  for(unsigned int i = 0; i < N; i++)
    gamma[T - 1][i] = alpha[T - 1][i] / (normalizer + EPSILON);
  }
  else if(type == LOG_SCALING)
  /* If alpha and beta are logarithmic, a different calculation of gamma and
     di-gamma must be done. It calculates the logarithm of them. */
  {
    for(unsigned int t = 0; t < T - 1; t++)
    {
      normalizer = LOGZERO;
      for(unsigned int i = 0; i < N; i++)
        for(unsigned int j = 0; j < N; j++)
        {
          diGamma[t][i][j] = elnproduct(alpha[t][i], elnproduct(eln(A[i][j]),
            elnproduct(eln(B[j][sequence[t + 1]]), beta[t + 1][j])));
          normalizer = elnsum(normalizer, diGamma[t][i][j]);
        }

      for(unsigned int i = 0; i < N; i++)
      {
        gamma[t][i] = LOGZERO;
        for(unsigned int j = 0; j < N; j++)
        {
          diGamma[t][i][j] =  elnproduct(diGamma[t][i][j],(-1.0 * normalizer));
          gamma[t][i] = elnsum(gamma[t][i], diGamma[t][i][j]);
        }
      }
    }

    /* Special computation for last time step (gamma) */
    normalizer = LOGZERO;
    for(unsigned int i = 0; i < N; i++)
      normalizer = elnsum(normalizer, alpha[T - 1][i]);
    for(unsigned int i = 0; i < N; i++)
      gamma[T - 1][i] = elnproduct(alpha[T - 1][i], (-1.0 * normalizer));
  }
}


/* Necessary functions for the logarithm treatment */

double eexp(double x)
{
  if(std::isnan(x))
    return 0.0;
  else
    return std::exp(x);
}

double eln(double x)
{
  if(x == 0.0)
    return LOGZERO;
  else
    return std::log(x);
}

double elnsum(double x, double y)
{
  if(std::isnan(x) || std::isnan(y))
    if(std::isnan(x))
      return y;
    else
      return x;
  else
    if(x > y)
      return (x + eln(1+eexp(y-x)));
    else
      return (y + eln(1+eexp(x-y)));
}

double elnproduct(double x, double y)
{
  if(std::isnan(x) || std::isnan(y))
    return LOGZERO;
  else
    return (x+y);
}

std::vector<double> normalizeProbDist(std::vector<double> v)
{
  double normalizer = 0.0;
  double rowSum = 0.0;

  for(unsigned int i = 0; i < v.size(); i++)
    normalizer += v[i];

  if(normalizer == 0.0)
    return v;

  for(unsigned int i = 0; i < v.size() - 1; i++)
  {
    v[i] /= (normalizer);
    rowSum += v[i];
  }
  v[v.size() - 1] = 1.0 - rowSum;

  return v;
}
