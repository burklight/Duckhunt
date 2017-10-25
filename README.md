* Duckhunt implementation 

** The assignement

This implementation was made to fulfill the DD2380 (Artificial Intellignece) first assignement. This assignement consisted in a
AI program able to predict the flight trajectory of several birds in order to shoot with high confidence. Furthermore, it was 
necessary to guess to which species those birds belonged to. This second objective was very important since shooting a Black Stork
would highly lower your punctuation. All these predictions shoud be done only given the observation of past movements.
For more information on the assignement and the skeleton of the code, please refer to [1](https://kth.kattis.com/problems/kth.ai.duckhunt).

** The code

In this repository you will be able to find 2 basic modules:

1. Player: In these files all the algorithms related to the prediction of next movements and species guessing can be found.
2. HMM: In these files the implementation of Hidden Markov Models can be found. In this implementation we have manly used two papers:
  - [A tutorial on Hidden Markov model and selected applications in speech recognition.](http://www.ece.ucsb.edu/Faculty/Rabiner/ece259/Reprints/tutorial%20on%20hmm%20and%20applications.pdf)
  - [A Revealing Introduction to Hidden Markov Models](https://www.cs.sjsu.edu/~stamp/RUA/HMM.pdf)
