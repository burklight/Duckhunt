#ifndef _DUCKS_PLAYER_HPP_
#define _DUCKS_PLAYER_HPP_

#include "Deadline.hpp"
#include "GameState.hpp"
#include "Action.hpp"
#include "HMM.hpp"
#include <vector>
#include <unordered_map>

namespace ducks
{

double getMedian(std::vector<double> v);
double getMean(std::vector<double> v);
double getEntropy(std::vector<double> v);
std::vector<int> concatenate(std::vector<int> v1, std::vector<int> v2, int lim);

class Player
{
public:
    /**
     * Constructor
     * There is no data in the beginning, so not much should be done here.
     */
    Player();

    /**
     * Shoot!
     *
     * This is the function where you start your work.
     *
     * You will receive a variable pState, which contains information about all
     * birds, both dead and alive. Each birds contains all past actions.
     *
     * The state also contains the scores for all players and the number of
     * time steps elapsed since the last time this function was called.
     *
     * @param pState the GameState object with observations etc
     * @param pDue time before which we must have returned
     * @return the prediction of a bird we want to shoot at, or cDontShoot to pass
     */
    Action shoot(const GameState &pState, const Deadline &pDue);

    /**
     * Guess the species!
     * This function will be called at the end of each round, to give you
     * a chance to identify the species of the birds for extra points.
     *
     * Fill the vector with guesses for the all birds.
     * Use SPECIES_UNKNOWN to avoid guessing.
     *
     * @param pState the GameState object with observations etc
     * @param pDue time before which we must have returned
     * @return a vector with guesses for all the birds
     */
    std::vector<ESpecies> guess(const GameState &pState, const Deadline &pDue);

    /**
     * If you hit the bird you were trying to shoot, you will be notified
     * through this function.
     *
     * @param pState the GameState object with observations etc
     * @param pBird the bird you hit
     * @param pDue time before which we must have returned
     */
    void hit(const GameState &pState, int pBird, const Deadline &pDue);

    /**
     * If you made any guesses, you will find out the true species of those
     * birds through this function.
     *
     * @param pState the GameState object with observations etc
     * @param pSpecies the vector with species
     * @param pDue time before which we must have returned
     */
    void reveal(const GameState &pState, const std::vector<ESpecies> &pSpecies, const Deadline &pDue);

private:
  /* Hidden variables of the class player */
  unsigned int successShot;   // Number of shots that hit a bird
  unsigned int totalShots;    // Number of shots fired
  unsigned int successGuess;  // Number of correct guesses
  unsigned int totalGuesses;  // Number of guesses done
  std::vector<ESpecies> guesses;  // Track of the guesses to see if they were right
  std::vector<std::vector<HMM> > birdModels; // HMM moels for all the birds
  std::vector<std::vector<int> > lastObservations; // Last observations of a given species


  /* Hidden functions of the class player */
  std::vector<int> getObservations(Bird b);
  std::vector<bool> revealedSpecies; // True for revealed species, false for unrevealed
  int getSpeciesGuess(std::vector<int> sequence, double &confidence, double& blackprob);
  int getKnownSpecies(); //get total number of revealed species
};

} /*namespace ducks*/

#endif
