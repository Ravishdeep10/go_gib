package go_gib

import (
	"bufio"
	"fmt"
	"log"
	"math"
	"os"
	"strings"
)

// Basic Tuple structure for rune i.e. char
type runeTuple struct{
	left  rune
	right rune
}

// The model we would like after training Markov Chain
type model struct {
	matrix [][]float64
	threshold float64
}

// Initialize a mapping all of the characters the alphabet to their numerical positon
var acceptedChars string = `abcdefghijklmnopqrstuvwxyz `
var alphabet map[rune]int

func init() {
	alphabet = make(map[rune]int)

	for pos, char := range acceptedChars {
		alphabet[char] = pos
	}
}

// A basic function to sum an array of float64
func sumArray(input []float64) float64 {
	var sum float64

	for i := range input {
		sum += input[i]
	}

	return sum
}

// A basic function to find the min an array of float64
func minFloatSlice(v []float64) (m float64) {
	if len(v) > 0 {
		m = v[0]
	}
	for i := 1; i < len(v); i++ {
		if v[i] < m {
			m = v[i]
		}
	}
	return
}

// A basic function to find the max an array of float64
func maxFloatSlice(v []float64) (m float64) {
	if len(v) > 0 {
		m = v[0]
	}
	for i := 1; i < len(v); i++ {
		if v[i] > m {
			m = v[i]
		}
	}
	return
}

// Thus function creates an array of all characters in a string
func normalize(line string) []rune {

	lineCharacters := make([]rune, 0)

	loweredString :=  strings.ToLower(line)

	for _, char := range loweredString {
		// Make sure the characters are not punctuations or infrequent symbols
		_, ok := alphabet[char]
		if ok {
			lineCharacters = append(lineCharacters, char)
		}
	}

	return lineCharacters
}

// Return 2 grams from the normalized character array
func doubeleGram(line string) []runeTuple{

	// An array of tuples containing runes
	grams := make([]runeTuple, 0)

	filtered := normalize(line)

	// Get every adjacent pair of characters in filtered as tuples in the array
	for i := 0; i < len(filtered) - 1; i++ {
		grams = append(grams, runeTuple{filtered[i], filtered[i+1]})
	}

	return grams
}


// This function trains a simple model
func train() {
	k := len(acceptedChars)

	// We will assume we have seen each character 10 times in order to give a prior probability
	priorFactor := make([]([]float64), k)

	for i := 0; i < k; i++ {
		vector := make([]float64, k)
		for j := 0; j < k; j++ {
			vector[j] = 10
		}
		priorFactor[i] = vector
	}

	// Count the transisitions of each character from "two_cities.txt"
	file, err := os.Open("two_cities.txt")
	if err != nil {
		log.Fatalln(err)
	}

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		transitions := doubeleGram(scanner.Text())
		for i := range transitions {
			a := transitions[i].left
			b := transitions[i].right
			priorFactor[alphabet[a]][alphabet[b]] += 1
		}
	}

	file.Close()

	if err := scanner.Err(); err != nil {
		log.Fatalln(err)
	}


	// Normalize the probabilities so that they become log probabilities

	for i := range priorFactor {
		s := sumArray(priorFactor[i])
		for j := range priorFactor[i] {
			priorFactor[i][j] = math.Log(priorFactor[i][j] / s)
		}
	}

	// Find the probability of generating a few good and bad phrases

	goodProbs := avgTransitionalProb(priorFactor, "english.txt")
	badProbs := avgTransitionalProb(priorFactor, "gibberish.txt")

	fmt.Println(minFloatSlice(goodProbs) > maxFloatSlice(badProbs))

	// Pick a threshold halfway between best bad and worst good probabilities
	threshold := (minFloatSlice(goodProbs) + maxFloatSlice(badProbs)) / 2.0

	fmt.Println(model{priorFactor, threshold})

}

// This function returns the average transition probabilities from a file through the logProbMatrix
func avgTransitionalProb(logProbMatrix [][]float64, filename string) []float64{


	var probabilities = make([]float64, 0)
	file, err := os.Open(filename)
	if err != nil {
		log.Fatalln(err)
	}

	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		probabilities = append(probabilities, avgTP(scanner.Text(), logProbMatrix))
	}

	if err := scanner.Err(); err != nil {
		log.Fatalln(err)
	}

	return probabilities
}

// This function computes the average transition probability for a line in the file
func avgTP(line string, logProbMat [][]float64) float64 {
	logProb := 0.0
	transCnt := 0
	transitions := doubeleGram(line)
	for i := range transitions {
		a := transitions[i].left
		b := transitions[i].right
		logProb += logProbMat[alphabet[a]][alphabet[b]]
		transCnt += 1
	}

	// We can not have a divisor of 0
	if transCnt == 0 {
		transCnt = 1
	}

	return math.Exp(logProb / float64(transCnt))
}
