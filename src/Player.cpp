#include "Player.hpp"
#include <cstdlib>
#include <iostream>
#include <algorithm>

#define MAXTURNS 99

namespace ducks
{

constexpr double GUESS_THRESHOLD = 0.5; // Threshold to determine if guessing probability is good enough (2 times it's probability at random)
constexpr double SHOOT_THRESHOLD = 0.5; // Threshold to determine if the probability is good enough for shooting for a single predictor model
constexpr double VOTING_THRESHOLD = 0.55; // Threshold to determine if enough models agree on shooting decision
constexpr double BLACK_TRHESHOLD = 1e-10; // Threshold to be confident enough of not haveing a black stork
constexpr unsigned int type = CONST_SCALING; // Scaling method for the HMM algorithms
constexpr unsigned int nStates = 3; // Number of hidden states (not 5 as we expected)

Player::Player()
{
   successShot = 0; // Number of successful shots (debbugging)
   totalShots = 0;  // Number of total shots (debbuging)
   successGuess = 0; // Number of successful guesses (debbuging)
   totalGuesses = 0;  // Number of total guesses (debbuging)
   birdModels.resize(COUNT_SPECIES); // Each species will have its own array of models
   revealedSpecies = std::vector<bool>(COUNT_SPECIES, false); // True for revealed species, false for unrevealed
   lastObservations.resize(COUNT_SPECIES);
}

Action Player::shoot(const GameState &pState, const Deadline &pDue)
{
  if(pState.getRound() == 0)
    return cDontShoot;

  double pre_shot = 1.5;
  if(pState.getNumPlayers() > 1)
    pre_shot = 3.0;

  unsigned int nBirds = pState.getNumBirds();   // Number of birds
  unsigned int startTime = MAXTURNS - (unsigned int)(pre_shot * nBirds);   // Turn to start shooting

  std::vector<int> observations; // Movements of each bird

  int birdSpecies = SPECIES_UNKNOWN;  // Species of the observed bird
  double confidence = 0.0;            // Confidence on the bird species
  double blackprob = 0.0;             // Probability of the bird being a BLACK STORK

  int targetBird = -1;
  int targetMove = -1;
  int targetSpecies = -1;
  double targetProbability = 0.0;

  for(unsigned int birdIndex = 0; birdIndex < nBirds; birdIndex++)
  {
    // We get the bird given by the pState
    Bird bird = pState.getBird(birdIndex);

    // We only act after the start time and only try to shoot birds that are alive
    if(bird.getSeqLength() < startTime || !bird.isAlive()) continue;

    // We get the movement of the bird
    observations = getObservations(bird);
    observations = concatenate(lastObservations[birdSpecies], observations, 350);
    birdSpecies = getSpeciesGuess(observations, confidence, blackprob);

    // We only try to shoot birds we know are not BLACK STORKS
    if(birdSpecies == SPECIES_UNKNOWN || blackprob > BLACK_TRHESHOLD) continue;

    // Create auxiliary vectors for statistics on the movement selection
    std::vector<std::vector<double> > xProbabilities(COUNT_MOVE);
    std::vector<double> moveProbabilites(COUNT_MOVE);
    std::vector<double> xMoves(COUNT_MOVE, 0.0);
	  double xProbability; int xMove;
    std::vector<double> hiddenStatesShotter = {2,4,5,6};

    // Number of models that we will evaluate
    unsigned int nModels = birdModels[birdSpecies].size()
      + hiddenStatesShotter.size();

    // We train different models with different number of hidden states
    // because they could model better the movement behaviour
    for(auto w : hiddenStatesShotter)
    {
      // We train a model with the current observations
      HMM model(w, COUNT_MOVE);
      model.estimateModel(observations, type);

      // We compute the most likely move for this model
      xProbability = 0.0;
      xMove = model.getNextMostProbObs(observations, xProbability);
      xMoves[xMove] += 1.0 / ((double) nModels);
      xProbabilities[xMove].push_back(xProbability);
    }

    // We do the same now for all the models stored for that bird species
    for(auto bModel : birdModels[birdSpecies])
    {
      xMove = bModel.getNextMostProbObs(observations, xProbability);
      xMoves[xMove] += 1.0 / ((double) nModels);
      xProbabilities[xMove].push_back(xProbability);
    }

    // We get the mean of the probabilies of each move given by the models
    // Then we normalize for all the possible moves
    for(unsigned int i = 0; i < COUNT_MOVE; i++)
      moveProbabilites[i] = getMean(xProbabilities[i]);
    moveProbabilites = normalizeProbDist(moveProbabilites);

    // We check if any movement is above the voting threshold
    for(unsigned int i = 0; i < COUNT_MOVE; i++)
    {
      if(xMoves[i] > VOTING_THRESHOLD
         && moveProbabilites[i] * confidence > targetProbability)
      {
        targetMove = i;
        targetProbability = moveProbabilites[i] * confidence;
        targetBird = birdIndex;
        targetSpecies = birdSpecies;
      }
    }

    // If this probability is above the shooting threshold we shoot
    if(targetProbability > SHOOT_THRESHOLD)
    {
      totalShots++;
      return Action(targetBird, EMovement(targetMove));
    }
  }

  return cDontShoot;

}

std::vector<ESpecies> Player::guess(const GameState &pState, const Deadline &pDue)
{
    int nBirds = pState.getNumBirds();
    std::vector<ESpecies> lGuesses(nBirds, SPECIES_UNKNOWN);

    /* On the first round we are forced to guess at random in order to get
       information about the birds species */
  	if (pState.getRound() == 0)
    {
		  for (unsigned int i = 0; i < nBirds; i++)
			   lGuesses[i] = ESpecies(SPECIES_PIGEON);
    }
    /* After the first round we can just guess normally */
    else
	  {
       Bird bird;
       int guess;
       std::vector<int> observations;
       double confidence = 0.0;
       double blackprob = 0.0;

      // For all the birds...
      for(unsigned int birdIndex = 0; birdIndex < nBirds; birdIndex++)
      {
  			bird = pState.getBird(birdIndex);
  			observations = getObservations(bird);
  			guess = getSpeciesGuess(observations, confidence, blackprob);
  			//Guess the species if the confidence of the guess is above the threshold
  			if(ESpecies(guess) != SPECIES_UNKNOWN && confidence > GUESS_THRESHOLD)
  				lGuesses[birdIndex] = ESpecies(guess);
  			else if(blackprob > BLACK_TRHESHOLD)
  				lGuesses[birdIndex] = ESpecies(SPECIES_BLACK_STORK);
        else
  			{
  				// Vote a random species from those that have not been revealed yet
  				unsigned int randIdx = std::rand() % COUNT_SPECIES; // random starting index
  				unsigned int stop = 0; // iteration counter
  				for (unsigned int i = randIdx; stop < COUNT_SPECIES; i = (i + 1) % COUNT_SPECIES)
  				{
  					if (!revealedSpecies[i] && (i != SPECIES_BLACK_STORK || getKnownSpecies() == COUNT_SPECIES - 1))
  					//if species has not been revealed before you can guess it
  					{
  						lGuesses[birdIndex] = ESpecies(i);
  						break;
  					}
  					stop++;
  				}
  			}
  		}
	  }
    guesses = lGuesses;

    return lGuesses;
}

void Player::hit(const GameState &pState, int pBird, const Deadline &pDue)
{
    /*
     * If you hit the bird you are trying to shoot, you will be notified through this function.
     */

     double confidence, blackprob;
     std::vector<int> observations = getObservations(pState.getBird(pBird));
     int birdSpecies = getSpeciesGuess(observations, confidence, blackprob);

  	 std::cerr << "HIT BIRD!!!" << std::endl;
  	 successShot++;
}

void Player::reveal(const GameState &pState, const std::vector<ESpecies> &pSpecies, const Deadline &pDue)
{
    /*
     * If you made any guesses, you will find out the true species of those birds in this function.
     */

     int nBirds = pSpecies.size();

     /* To know if we are guessing right... */
     for(unsigned int i = 0; i < nBirds; i++)
     {
       if(pSpecies[i] == guesses[i])
         successGuess++;
       totalGuesses++;
     }

     /* Train the bird models knowing it's species, then store it for better guessing */
     std::vector<int> observations;
     for(unsigned int birdIndex = 0; birdIndex < nBirds; birdIndex++)
     {
		    if (pSpecies[birdIndex] == -1)
				continue;
    		HMM model(nStates, COUNT_MOVE);
    		observations = getObservations(pState.getBird(birdIndex));
    		model.estimateModel(observations, CONST_SCALING);
    		birdModels[pSpecies[birdIndex]].push_back(model);
    		revealedSpecies[pSpecies[birdIndex]] = true;
        /* Also store the last sequence of observations for each species to shoot better */
        lastObservations[pSpecies[birdIndex]] =
          concatenate(lastObservations[pSpecies[birdIndex]],observations,250);
     }


     // The below cerrs are just for debbuging
     std::cerr << "\nRound #" << pState.getRound();

     std::cerr << "\nNum of guesses " << totalGuesses;
     std::cerr << "\nNum of shots: " << totalShots;

     if(totalGuesses == 0)
       std::cerr << "\nGuess ratio: 0";
     else
       std::cerr << "\n\nGuess ratio: " << ((double) successGuess)/((double) totalGuesses);
     if(totalShots == 0)
       std::cerr << "\nShoot ratio: 0";
     else
       std::cerr << "\nShoot ratio: " << ((double) successShot)/((double) totalShots);
     std::cerr << "\n\n";

     successGuess = 0;
     totalGuesses = 0;
     successShot = 0;
     totalShots = 0;
}

std::vector<int> Player::getObservations(Bird b)
/* Returns the sequence of movements of a bird */
{
  unsigned int T = b.getSeqLength();
  std::vector<int> sequence(T);

  for(unsigned int t = 0; t < T; t++)
    if(b.wasAlive(t))
      sequence[t] = b.getObservation(t);

  return sequence;
}

int Player::getSpeciesGuess(std::vector<int> sequence, double &confidence,
  double &blackprob)
/* Returns a guess of the spice of a bird */
{
	int guess = SPECIES_UNKNOWN;
	double probability;
	double bestProbability = 0.0;
	std::vector<std::vector<double> > probabilities(COUNT_SPECIES, std::vector<double> (0)); //sum of probabilities for each model
	std::vector<double> speciesProbabilities(COUNT_SPECIES, 0.0);

	// For each species
	for(unsigned int i = 0; i < COUNT_SPECIES; i++)
	{
		// For each model trained for this species
		for(unsigned int j = 0; j < birdModels[i].size(); j++)
		{
			// get the observation probability given the model parameters, P(O|lamda)
			probability = eexp(birdModels[i][j].getObsSeqProb(sequence, type));
			probabilities[i].push_back(probability);
		}
	}

  // We take the mean of the probabilities given by the models
	for(unsigned int i = 0; i < COUNT_SPECIES; i++)
		speciesProbabilities[i] = getMean(probabilities[i]);

  // We normalize the probabilities (P(A|B) = alpha * P(A,B))
	speciesProbabilities = normalizeProbDist(speciesProbabilities);

	bestProbability = 0.0;
	// Get the max probability from the distribution
	for(int i = 0; i < COUNT_SPECIES; i++)
	{
		probability = speciesProbabilities[i];
		if(probability > bestProbability)
		{
			guess = i;
			bestProbability = probability;
		}
	}

  // We make sure to put confidence to 0 if we don't know the species
	if(guess == SPECIES_UNKNOWN)
		confidence = 0.0;
  else
		confidence = speciesProbabilities[guess];

  blackprob = speciesProbabilities[SPECIES_BLACK_STORK];

	return ESpecies(guess);
}

int Player::getKnownSpecies()
{
	int sum = 0;
	for (int i = 0; i < revealedSpecies.size(); i++)
		if (revealedSpecies[i] == true)
			sum++;
	return sum;
}


// Additional useful functions

double getMedian(std::vector<double> v)
{
  double median;
  unsigned int size = v.size();

  if(size == 0)
    return 0.0;

  std::sort(v.begin(), v.end());

  if (size  % 2 == 0)
    median = (v[size / 2 - 1] + v[size / 2]) / 2;
  else
    median = v[size / 2];

  return median;
}

double getMean(std::vector<double> v)
{
  double mean = 0.0;
  unsigned int size = v.size();

  if(size == 0)
    return 0.0;

  for(unsigned int i = 0; i < size; i++)
    mean += v[i];

  mean /= size;

  return mean;
}

double getEntropy(std::vector<double> v)
{
  double entropy = 0.0;

  for(unsigned int i = 0; i < v.size(); i++)
    entropy -= v[i] * eln(v[i]);

  return entropy;
}

std::vector<int> concatenate(std::vector<int> v1, std::vector<int> v2, int lim)
{
  int N = v1.size();
  int M = v2.size();
  std::vector<int> w(std::min(N+M, lim));

  if(lim <= N)
    for(unsigned int i = 0; i < lim; i++)
      w[i] = v1[i];
  else if(lim < N + M)
  {
    for(unsigned int i = 0; i < N; i++)
      w[i] = v1[i];
    for(unsigned int j = 0; j < (lim - N); j++)
      w[N + j] = v2[j];
  }
  else
  {
    for(unsigned int i = 0; i < N; i++)
      w[i] = v1[i];
    for(unsigned int j = 0; j < M; j++)
      w[N + j] = v2[j];
  }

  return w;
}

} /*namespace ducks*/
