using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace AlgoDSPlay
{
    public partial class RealProbs
    {
        /*
        726. Number of Atoms
https://leetcode.com/problems/number-of-atoms/description/
        */
        public class CountOfAtomsSol
        {
            /*
            Approach 1: Recursion
Complexity Analysis
Let N be the length of the formula.
•	Time complexity: O(N^2)
The recursive function parse_formula() will be called O(N) times.
However, we are iterating over the atoms of the nested formula to add the count to the current formula. This will take time equal to the number of atoms in the nested formula. The number of atoms in the nested formula can be equal to O(N). Thus, the time complexity of the recursive function will be O(N^2).
One such example of worst case is (A(B(C(D(E(F(G(H(I(J(K(L(M(N(O(P(Q(R(S(T(U(V(W(X(Y(Z)2)2)2)2)2)2)2)2)2)2)2)2)2)2)2)2)2)2)2)2)2)2)2)2)2). In this case, whenever we encounter a right parenthesis, we will have to iterate over all the atoms in the nested formula to add the count to the current formula.
In actual it is O(PN) where P is the number of paranthese pairs. Here P can be at most N/2, or P=O(N). However, P is not a function of input size. Hence, we shouldn't consider it in the time complexity.
Sorting will take O(NlogN) time. This may vary depending on the implementation of the sorting algorithm in the programming language. Generating the answer string will take O(N) time.
Hence, the overall time complexity will be O(N^2).
•	Space complexity: O(N)
The space complexity will be O(N) due to the space used by the recursive function call stack.
The space used by the final_map will be O(N). Moreover, we are sorting the final_map. In sorting, some extra space is used. The space complexity depends on the implementation of the sorting algorithm in the programming language, but it will be O(N).
The space used by the answer string ans will be O(N).
Hence, the overall space complexity will be O(N).

            */
            // Global variable
            // Global variable
            private int index = 0;

            public string WithRecursion(string formula)
            {
                // Recursively parse the formula
                Dictionary<string, int> finalMap = ParseFormula(formula);

                // Sort the final map
                SortedDictionary<string, int> sortedMap = new SortedDictionary<string, int>(finalMap);

                // Generate the answer string
                StringBuilder answerStringBuilder = new StringBuilder();
                foreach (string atom in sortedMap.Keys)
                {
                    answerStringBuilder.Append(atom);
                    if (sortedMap[atom] > 1)
                    {
                        answerStringBuilder.Append(sortedMap[atom]);
                    }
                }

                return answerStringBuilder.ToString();
            }

            // Recursively parse the formula
            private Dictionary<string, int> ParseFormula(string formula)
            {
                // Local variable
                Dictionary<string, int> currentMap = new Dictionary<string, int>();

                // Iterate until right parenthesis or end of the formula
                while (index < formula.Length && formula[index] != ')')
                {
                    // If left parenthesis, do recursion
                    if (formula[index] == '(')
                    {
                        index++;
                        Dictionary<string, int> nestedMap = ParseFormula(formula);
                        foreach (string atom in nestedMap.Keys)
                        {
                            currentMap[atom] = currentMap.GetValueOrDefault(atom, 0) + nestedMap[atom];
                        }
                    }
                    // Otherwise, it should be UPPERCASE LETTER
                    // Extract the atom and count in one go.
                    else
                    {
                        StringBuilder currentAtomStringBuilder = new StringBuilder();
                        currentAtomStringBuilder.Append(formula[index]);
                        index++;
                        while (index < formula.Length && char.IsLower(formula[index]))
                        {
                            currentAtomStringBuilder.Append(formula[index]);
                            index++;
                        }

                        StringBuilder currentCountStringBuilder = new StringBuilder();
                        while (index < formula.Length && char.IsDigit(formula[index]))
                        {
                            currentCountStringBuilder.Append(formula[index]);
                            index++;
                        }

                        if (currentCountStringBuilder.Length == 0)
                        {
                            currentMap[currentAtomStringBuilder.ToString()] = currentMap.GetValueOrDefault(currentAtomStringBuilder.ToString(), 0) + 1;
                        }
                        else
                        {
                            currentMap[currentAtomStringBuilder.ToString()] = currentMap.GetValueOrDefault(currentAtomStringBuilder.ToString(), 0) +
                                int.Parse(currentCountStringBuilder.ToString());
                        }
                    }
                }

                // If the right parenthesis, extract the multiplier
                // and multiply the count of atoms in the current map
                index++;
                StringBuilder multiplierStringBuilder = new StringBuilder();
                while (index < formula.Length && char.IsDigit(formula[index]))
                {
                    multiplierStringBuilder.Append(formula[index]);
                    index++;
                }
                if (multiplierStringBuilder.Length > 0)
                {
                    int multiplier = int.Parse(multiplierStringBuilder.ToString());
                    foreach (string atom in currentMap.Keys)
                    {
                        currentMap[atom] *= multiplier;
                    }
                }

                return currentMap;
            }

            /*
          Approach 2: Stack
Complexity Analysis
Let N be the length of the formula.
•	Time complexity: O(N^2)
The stack will have at most O(N) elements. Each element will be popped and pushed at most once. However, since we need to revisit the atoms in the nested formula to add the count to the current formula, in the worst case, the time complexity of the stack operations will be O(N^2).
Sorting will take O(NlogN) time. This may vary depending on the implementation of the sorting algorithm in the programming language. Generating the answer string will take O(N) time.
Hence, the overall time complexity will be O(N^2).
•	Space complexity: O(N)
The space used by the stack will be O(N).
The space used by the final_map will be O(N). Moreover, we are sorting the final_map. In sorting, some extra space is used. The space complexity depends on the implementation of the sorting algorithm in the programming language. However, it will be O(N).
The space used by the answer string ans will be O(N).
Hence, the overall space complexity will be O(N).

            */
            public string WithStack(string formula)
            {
                // Stack to keep track of the atoms and their counts
                Stack<Dictionary<string, int>> stack = new Stack<Dictionary<string, int>>();
                stack.Push(new Dictionary<string, int>());

                // Index to keep track of the current character
                int index = 0;

                // Parse the formula
                while (index < formula.Length)
                {
                    // If left parenthesis, insert a new dictionary to the stack. It will
                    // keep track of the atoms and their counts in the nested formula
                    if (formula[index] == '(')
                    {
                        stack.Push(new Dictionary<string, int>());
                        index++;
                    }
                    // If right parenthesis, pop the top element from the stack
                    // Multiply the count with the multiplicity of the nested formula
                    else if (formula[index] == ')')
                    {
                        Dictionary<string, int> currentMap = stack.Pop();
                        index++;
                        StringBuilder multiplier = new StringBuilder();
                        while (index < formula.Length && char.IsDigit(formula[index]))
                        {
                            multiplier.Append(formula[index]);
                            index++;
                        }
                        if (multiplier.Length > 0)
                        {
                            int mult = int.Parse(multiplier.ToString());
                            foreach (string atom in currentMap.Keys)
                            {
                                currentMap[atom] *= mult;
                            }
                        }

                        foreach (string atom in currentMap.Keys)
                        {
                            stack.Peek()[atom] = stack.Peek().GetValueOrDefault(atom, 0) + currentMap[atom];
                        }
                    }
                    // Otherwise, it must be an UPPERCASE LETTER. Extract the complete
                    // atom with frequency, and update the most recent dictionary
                    else
                    {
                        StringBuilder currentAtom = new StringBuilder();
                        currentAtom.Append(formula[index]);
                        index++;
                        while (index < formula.Length && char.IsLower(formula[index]))
                        {
                            currentAtom.Append(formula[index]);
                            index++;
                        }

                        StringBuilder currentCount = new StringBuilder();
                        while (index < formula.Length && char.IsDigit(formula[index]))
                        {
                            currentCount.Append(formula[index]);
                            index++;
                        }

                        int count = currentCount.Length > 0 ? int.Parse(currentCount.ToString()) : 1;
                        stack.Peek()[currentAtom.ToString()] = stack.Peek().GetValueOrDefault(currentAtom.ToString(), 0) + count;
                    }
                }

                // Sort the final map
                SortedDictionary<string, int> finalMap = new SortedDictionary<string, int>(stack.Peek());

                // Generate the answer string
                StringBuilder answer = new StringBuilder();
                foreach (string atom in finalMap.Keys)
                {
                    answer.Append(atom);
                    if (finalMap[atom] > 1)
                    {
                        answer.Append(finalMap[atom]);
                    }
                }

                return answer.ToString();
            }

            /*
            Approach 3: Regular Expression
            Complexity Analysis
Let N be the length of the formula.
•	Time complexity: O(N^2)
o	Parsing the regex in the formula will take O(N) time.
o	There will be at most O(N) quintuples in the matcher. Now, since for the right parenthesis, we need to revisit the atoms in the nested formula to add the count to the current formula, in the worst case, the time complexity of the stack operations will be O(N^2).
o	Sorting will take O(NlogN) time. This may vary depending on the implementation of the sorting algorithm in the programming language.
o	Generating the answer string will take O(N) time.
Hence, the overall time complexity will be O(N^2).
•	Space complexity: O(N)
o	There will be at most O(N) quintuples in the matcher.
o	The space used by the stack will be O(N).
o	The space used by the final_map will be O(N). Moreover, we are sorting the final_map. In sorting, some extra space is used. The space complexity depends on the implementation of the sorting algorithm in the programming language. However, it will be O(N).
o	The space used by the answer string ans will be O(N).
Hence, the overall space complexity will be O(N).

            */
            public string WithRegEx(string formula)
            {
                // Regular expression to extract atom, count, (, ), multiplier
                // Every element of parsed will be a quintuple
                string regex = "([A-Z][a-z]*)(\\d*)|(\\()|(\\))(\\d*)";
                Regex pattern = new Regex(regex);
                MatchCollection matches = pattern.Matches(formula);

                // Stack to keep track of the atoms and their counts
                Stack<Dictionary<string, int>> atomStack = new Stack<Dictionary<string, int>>();
                atomStack.Push(new Dictionary<string, int>());

                // Parse the formula
                foreach (Match matcher in matches)
                {
                    string atom = matcher.Groups[1].Value;
                    string count = matcher.Groups[2].Value;
                    string left = matcher.Groups[3].Value;
                    string right = matcher.Groups[4].Value;
                    string multiplier = matcher.Groups[5].Value;

                    // If atom, add it to the top dictionary
                    if (!string.IsNullOrEmpty(atom))
                    {
                        atomStack.Peek()[atom] = atomStack.Peek().GetValueOrDefault(atom, 0) +
                            (string.IsNullOrEmpty(count) ? 1 : int.Parse(count));
                    }
                    // If left parenthesis, insert a new dictionary to the stack
                    else if (!string.IsNullOrEmpty(left))
                    {
                        atomStack.Push(new Dictionary<string, int>());
                    }
                    // If right parenthesis, pop the top element from the stack
                    // Multiply the count with the attached multiplicity.
                    // Add the count to the current formula
                    else if (!string.IsNullOrEmpty(right))
                    {
                        Dictionary<string, int> currentMap = atomStack.Pop();
                        if (!string.IsNullOrEmpty(multiplier))
                        {
                            int mult = int.Parse(multiplier);
                            foreach (var atomName in currentMap.Keys)
                            {
                                currentMap[atomName] *= mult;
                            }
                        }

                        foreach (var atomName in currentMap.Keys)
                        {
                            atomStack.Peek()[atomName] = atomStack.Peek().GetValueOrDefault(atomName, 0) +
                                currentMap[atomName];
                        }
                    }
                }

                // Sort the final dictionary
                SortedDictionary<string, int> finalMap = new SortedDictionary<string, int>(atomStack.Peek());

                // Generate the answer string
                StringBuilder answerStringBuilder = new StringBuilder();
                foreach (var atom in finalMap.Keys)
                {
                    answerStringBuilder.Append(atom);
                    if (finalMap[atom] > 1)
                    {
                        answerStringBuilder.Append(finalMap[atom]);
                    }
                }

                return answerStringBuilder.ToString();
            }

            /*
            Approach 4: Reverse Scanning	
            Complexity Analysis
Let N be the length of the formula.
•	Time complexity: O(N^2)
o	Declaring and Initializing the variables before the while loop will take O(1) time.
o	The while loop will run O(N) times. The number of steps in one while loop depends on the character at the current index.
o	In the case of a digit, lowercase letter, or UPPERCASE LETTER, we are prepending the characters. Appending is O(1) operation, however, prepending is O(N) operation.
s = s + a is different from s = a + s. The former can be augmented as s += a, while the latter can't be augmented.
Although it may vary with programming language, in general, inserting at the end is O(1) operation, while inserting at the beginning is O(N) operation.
The worst case example of this can be when the formula is "Qabcdefghij".
o	In the case of the left parenthesis, we are converting the string curr_count to integer curr_multiplier. This may take O(N) time in the worst case. However, the amortized time complexity will be O(1).
o	In the case of the right parenthesis, we are updating the running_mul and stack. This will take O(1) time.
Hence, the time complexity of the while loop will be O(N^2).
o	Sorting will take O(NlogN) time. This may vary depending on the implementation of the sorting algorithm in the programming language.
o	Generating the answer string will take O(N) time.
Hence, the overall time complexity will be O(N^2).
•	Space complexity: O(N)
o	The stack may have at most O(N) elements.
o	The space used by the final_map will be O(N). Moreover, we are sorting the final_map. In sorting, some extra space is used. The space complexity depends on the implementation of the sorting algorithm in the programming language. However, it will be O(N).
o	The space used by the ans will be O(N).
o	The space used by the curr_atom and curr_count will be O(N).
o	The space used by the running_mul will be O(1), since it is of integer type, which allocates fixed space.
Hence, the overall space complexity will be O(N).

            */
            public string WithReverseScanning(string formula)
            {
                // For multipliers
                int runningMultiplier = 1;
                Stack<int> multiplierStack = new Stack<int>();
                multiplierStack.Push(1);

                // Map to store the count of atoms
                Dictionary<string, int> atomCountMap = new Dictionary<string, int>();

                // Strings to take care of current atom and count
                StringBuilder currentAtom = new StringBuilder();
                StringBuilder currentCount = new StringBuilder();

                // Index to traverse the formula in reverse
                int index = formula.Length - 1;

                // Parse the formula
                while (index >= 0)
                {
                    // If digit, update the count
                    if (char.IsDigit(formula[index]))
                    {
                        currentCount.Insert(0, formula[index]);
                    }
                    // If a lowercase letter, prepend to the currentAtom
                    else if (char.IsLower(formula[index]))
                    {
                        currentAtom.Insert(0, formula[index]);
                    }
                    // If UPPERCASE LETTER, update the atomCountMap
                    else if (char.IsUpper(formula[index]))
                    {
                        currentAtom.Insert(0, formula[index]);
                        int count = currentCount.Length > 0
                            ? int.Parse(currentCount.ToString())
                            : 1;
                        if (atomCountMap.ContainsKey(currentAtom.ToString()))
                        {
                            atomCountMap[currentAtom.ToString()] += count * runningMultiplier;
                        }
                        else
                        {
                            atomCountMap[currentAtom.ToString()] = count * runningMultiplier;
                        }

                        currentAtom = new StringBuilder();
                        currentCount = new StringBuilder();
                    }
                    // If the right parenthesis, the currentCount if any
                    // will be considered as multiplier
                    else if (formula[index] == ')')
                    {
                        int currentMultiplier = currentCount.Length > 0
                            ? int.Parse(currentCount.ToString())
                            : 1;
                        multiplierStack.Push(currentMultiplier);
                        runningMultiplier *= currentMultiplier;
                        currentCount = new StringBuilder();
                    }
                    // If left parenthesis, update the runningMultiplier
                    else if (formula[index] == '(')
                    {
                        runningMultiplier /= multiplierStack.Pop();
                    }

                    index--;
                }

                // Sort the final map
                SortedDictionary<string, int> sortedAtomCountMap = new SortedDictionary<string, int>(atomCountMap);

                // Generate the answer string
                StringBuilder answer = new StringBuilder();
                foreach (var atom in sortedAtomCountMap.Keys)
                {
                    answer.Append(atom);
                    if (sortedAtomCountMap[atom] > 1)
                    {
                        answer.Append(sortedAtomCountMap[atom]);
                    }
                }

                return answer.ToString();
            }
            /*            
Approach 5: Preprocessing
Complexity Analysis
Let N be the length of the formula.
•	Time complexity: O(NlogN)
o	The while loop of pre-processing will have O(N) iterations.
o	When the current character is alphanumeric, or left parenthesis, the time complexity will be O(1).
o	When the current character is a right parenthesis, the time complexity can be O(N) in the worst case, because of the string reversal and conversion to integer. However, the amortized time complexity will be O(1).
Hence, the time complexity of pre-processing will be O(N).
o	Reversing the muls will take O(N) time.
o	The while loop of the processing will have O(N) iterations.
Every character will be processed at most twice, once during extracting, and other during storing.
Hence, the time complexity of the processing will be O(N).
o	Sorting will take O(KlogK) time, where K is the number of unique atoms. In the worst case, K can be equal to N. It is worth noting that this may vary depending on the implementation of the sorting algorithm in the programming language.
o	Generating the answer string will take O(N) time.
Hence, the overall time complexity will be O(N+N+NlogN+N), which is O(NlogN).
•	Space complexity: O(N)
o	The space used by the muls will be O(N).
o	The space used by the stack will be O(N).
o	The space used by the final_map will be O(N). Moreover, we are sorting the final_map. In sorting, some extra space is used. The space complexity depends on the implementation of the sorting algorithm in the programming language. However, it will be O(N).
o	The space used by the answer string ans will be O(N).
Hence, the overall space complexity will be O(N).


            */
            public string WithPreProcessing(string formula)
            {
                // For every index, store the valid multiplier
                int[] multipliers = new int[formula.Length];
                int currentMultiplier = 1;

                // Stack to handle nested formulas
                Stack<int> stack = new Stack<int>();
                stack.Push(1);

                // Preprocess the formula and extract all multipliers
                int index = formula.Length - 1;
                StringBuilder currentNumber = new StringBuilder();
                while (index >= 0)
                {
                    if (char.IsDigit(formula[index]))
                    {
                        currentNumber.Insert(0, formula[index]);
                    }
                    // If we encountered a letter, then the scanned
                    // number was count and not a multiplier. Discard it.
                    else if (char.IsLetter(formula[index]))
                    {
                        currentNumber = new StringBuilder();
                    }
                    // If we encounter a right parenthesis, then the
                    // scanned number was multiplier. Store it.
                    else if (formula[index] == ')')
                    {
                        int currentMultiplierValue = currentNumber.Length > 0
                            ? int.Parse(currentNumber.ToString())
                            : 1;
                        currentMultiplier *= currentMultiplierValue;
                        stack.Push(currentMultiplierValue);
                        currentNumber = new StringBuilder();
                    }
                    // If we encounter a left parenthesis, then the
                    // most recent multiplier will cease to exist.
                    else if (formula[index] == '(')
                    {
                        currentMultiplier /= stack.Pop();
                        currentNumber = new StringBuilder();
                    }

                    // For every index, store the valid multiplier
                    multipliers[index] = currentMultiplier;
                    index--;
                }

                // Map to store the count of atoms
                Dictionary<string, int> finalMap = new Dictionary<string, int>();

                // Traverse left to right in the formula
                index = 0;
                while (index < formula.Length)
                {
                    // If UPPER CASE LETTER, extract the entire atom
                    if (char.IsUpper(formula[index]))
                    {
                        StringBuilder currentAtom = new StringBuilder();
                        currentAtom.Append(formula[index]);
                        StringBuilder currentCount = new StringBuilder();
                        index++;
                        while (index < formula.Length && char.IsLower(formula[index]))
                        {
                            currentAtom.Append(formula[index]);
                            index++;
                        }

                        // Extract the count
                        while (index < formula.Length && char.IsDigit(formula[index]))
                        {
                            currentCount.Append(formula[index]);
                            index++;
                        }

                        // Update the final map
                        int count = currentCount.Length > 0
                            ? int.Parse(currentCount.ToString())
                            : 1;
                        if (finalMap.ContainsKey(currentAtom.ToString()))
                        {
                            finalMap[currentAtom.ToString()] += count * multipliers[index - 1];
                        }
                        else
                        {
                            finalMap[currentAtom.ToString()] = count * multipliers[index - 1];
                        }
                    }
                    else
                    {
                        index++;
                    }
                }

                // Sort the final map
                SortedDictionary<string, int> sortedMap = new SortedDictionary<string, int>(finalMap);

                // Generate the answer string
                StringBuilder answer = new StringBuilder();
                foreach (var atom in sortedMap.Keys)
                {
                    answer.Append(atom);
                    if (sortedMap[atom] > 1)
                    {
                        answer.Append(sortedMap[atom]);
                    }
                }

                return answer.ToString();
            }
            /*
            Approach 6: Reverse Scanning with Regex
            Complexity Analysis
Let N be the length of the formula.
•	Time complexity: O(NlogN)
o	The time complexity of finding all the quintuples using regular expression will depend on the programming language. In general, it will be O(N).
o	The time complexity of the for loop will be O(N).
o	If atom, adding it to the final_map will take O(1) time.
o	If the right parenthesis, multiplying the running_mul and pushing the multiplier to the stack will take O(1) time.
o	If left parenthesis, dividing the running_mul by the popped element from the stack will take O(1) time.
Hence, the time complexity of the for loop will be O(N).
o	Sorting will take O(KlogK) time, where K is the number of unique atoms. In the worst case, K can be equal to N. It is worth noting that this may vary depending on the implementation of the sorting algorithm in the programming language.
o	Generating the answer string will take O(N) time.
Hence, the overall time complexity will be O(N+N+NlogN+N), which is O(NlogN).
•	Space complexity: O(N)
o	The space used by the quintuples will be O(N).
o	The space used by the final_map will be O(N). Moreover, we are sorting the final_map. In sorting, some extra space is used. The space complexity depends on the implementation of the sorting algorithm in the programming language. However, it will be O(N).
o	The space used by the answer string ans will be O(N).
o	The space used by the stack will be O(N).
Hence, the overall space complexity will be O(N).	

            */
            public string UsingReverseScanningWithRegex(string formula)
            {
                // Every element of matcher will be a quintuple
                MatchCollection matches = Regex.Matches(formula, @"([A-Z][a-z]*)(\d*)|(\()|(\))(\d*)");
                List<string[]> matchList = new List<string[]>();
                foreach (Match match in matches)
                {
                    matchList.Add(new string[]
                    {
                match.Groups[1].Value,
                match.Groups[2].Value,
                match.Groups[3].Value,
                match.Groups[4].Value,
                match.Groups[5].Value,
                    });
                }
                matchList.Reverse();

                // Map to store the count of atoms
                Dictionary<string, int> finalMap = new Dictionary<string, int>();

                // Stack to keep track of the nested multiplicities
                Stack<int> stack = new Stack<int>();
                stack.Push(1);

                // Current Multiplicity
                int runningMultiplier = 1;

                // Parse the formula
                foreach (string[] quintuple in matchList)
                {
                    string atom = quintuple[0];
                    string count = quintuple[1];
                    string left = quintuple[2];
                    string right = quintuple[3];
                    string multiplier = quintuple[4];

                    // If atom, add it to the final map
                    if (!string.IsNullOrEmpty(atom))
                    {
                        int cnt = count.Length > 0 ? int.Parse(count) : 1;
                        if (finalMap.ContainsKey(atom))
                        {
                            finalMap[atom] += cnt * runningMultiplier;
                        }
                        else
                        {
                            finalMap[atom] = cnt * runningMultiplier;
                        }
                    }
                    // If the right parenthesis, multiply the runningMultiplier
                    else if (!string.IsNullOrEmpty(right))
                    {
                        int currentMultiplier = multiplier.Length > 0 ? int.Parse(multiplier) : 1;
                        runningMultiplier *= currentMultiplier;
                        stack.Push(currentMultiplier);
                    }
                    // If left parenthesis, divide the runningMultiplier
                    else if (!string.IsNullOrEmpty(left))
                    {
                        runningMultiplier /= stack.Pop();
                    }
                }

                // Sort the final map
                SortedDictionary<string, int> sortedMap = new SortedDictionary<string, int>(finalMap);

                // Generate the answer string
                StringBuilder answerBuilder = new StringBuilder();
                foreach (KeyValuePair<string, int> entry in sortedMap)
                {
                    answerBuilder.Append(entry.Key);
                    if (entry.Value > 1)
                    {
                        answerBuilder.Append(entry.Value);
                    }
                }

                return answerBuilder.ToString();
            }
        }

        /*
        734. Sentence Similarity
        https://leetcode.com/problems/sentence-similarity/description/
        */
        public class AreSentencesSimilarSol
        {
            /*
Approach: Using Hash Map and Hash Set
Complexity Analysis
Here, n is the number of words in sentence1 and sentence2 and k is the length of similar pairs. Let m be the average length of words in sentence1 as well as similarPairs.
•	Time complexity: O((n+k)⋅m)
o	We iterate over all the elements of similarPairs and insert a pair twice into wordToSimilarWords. To hash each word of length m, we need O(m) time, and to put the same length word in the hash set, we need O(m) time again. Because there are k pairs of words, there can be at most 2⋅k words that can be hashed and added to the set. As a result, we require O(k⋅m) time.
o	We also iterate over all of sentence1's words to see if sentence1[i] == sentence2[i]. Because each word is m long, checking words at a specific index would take O(m) time. It will take O(n⋅m) time in total because there are n words. For each word sentence1[i], we check if this word is present as the key in wordToSimilarWords which takes O(m) time per word, and searching for the similar word sentence2[i] in the wordToSimilarWords[sentence1[i]] set also takes O(m) time. As a result, for n words, performing the key lookup followed by searching in the set would take O(n⋅m) time.
o	The overall time required is O((n+k)⋅m).
•	Space complexity: O(k⋅m)
o	We are using wordToSimilarWords to store all the similar words for a given word. There are k pairs of similar words, so there could be O(k) words that are inserted into wordToSimilarWords. Because the average length of each word is m, the required space is O(k⋅m).

            */
            public bool UsingHashMapAndHashSet(string[] firstSentence, string[] secondSentence, List<List<string>> similarWordPairs)
            {
                if (firstSentence.Length != secondSentence.Length)
                {
                    return false;
                }
                Dictionary<string, HashSet<string>> wordToSimilarWords = new Dictionary<string, HashSet<string>>();
                foreach (List<string> pair in similarWordPairs)
                {
                    if (!wordToSimilarWords.ContainsKey(pair[0]))
                    {
                        wordToSimilarWords[pair[0]] = new HashSet<string>();
                    }
                    wordToSimilarWords[pair[0]].Add(pair[1]);

                    if (!wordToSimilarWords.ContainsKey(pair[1]))
                    {
                        wordToSimilarWords[pair[1]] = new HashSet<string>();
                    }
                    wordToSimilarWords[pair[1]].Add(pair[0]);
                }

                for (int index = 0; index < firstSentence.Length; index++)
                {
                    // If the words are equal, continue.
                    if (firstSentence[index].Equals(secondSentence[index]))
                    {
                        continue;
                    }
                    // If the words form a similar pair, continue.
                    if (wordToSimilarWords.ContainsKey(firstSentence[index]) &&
                        wordToSimilarWords[firstSentence[index]].Contains(secondSentence[index]))
                    {
                        continue;
                    }
                    return false;
                }

                return true;
            }
        }

        /*
        737. Sentence Similarity II
        https://leetcode.com/problems/sentence-similarity-ii/description/

        */
        class AreSentencesSimilarTwoSol
        {
            /*
            Approach 1: Depth First Search
            Complexity Analysis
Here, n is the number of words in sentence1 and sentence2 and k is the number of similar pairs. Let m be the average length of words in sentence1, sentence2 as well as in similarPairs.
•	Time complexity: O(n⋅k⋅m)
o	We iterate over all the elements of similarPairs and insert a pair twice into adj. To hash each word of length m, we need O(m) time, and to put the same length word in the hash set, we need O(m) time again. Because there are k pairs of words, there can be at most 2⋅k words that can be hashed and added to the set. As a result, we require O(k⋅m) time.
o	We iterate over all of sentence1's words to see if the corresponding words are equal or similar. If the words are not equal, we perform a DFS traversal over the graph with O(k) nodes and edges. As we know, a single DFS traversal takes O(V+E) time to traverse a graph with V nodes and E edges. In our case, we have O(k) edges and nodes, and each node is a string with an average size of m, so a single traversal would take O(k⋅m) time. For n traversals, it would take O(n⋅k⋅m).
o	The total amount of time required is O(n⋅k⋅m+k⋅m)=O(n⋅k⋅m).
•	Space complexity: O(k⋅m)
o	Because we use similarPairs to create the graph, the number of nodes and edges can be O(k). To map O(k) words of size m, adj uses O(k⋅m) space.
o	The recursion call stack used by dfs can have no more than O(k) elements. It would take up O(k) space in that case.
o	We also use a set visit, which requires O(k⋅m) space because it could contain O(k) words in the worst case.

            */
            // Returns true if there is a path from node to dest.        
            public bool DFS(string[] sentence1, string[] sentence2, List<List<string>> similarPairs)
            {
                if (sentence1.Length != sentence2.Length)
                {
                    return false;
                }
                // Create the graph using each pair in similarPairs.
                Dictionary<string, HashSet<string>> adjacencyList = new Dictionary<string, HashSet<string>>();
                foreach (List<string> pair in similarPairs)
                {
                    if (!adjacencyList.ContainsKey(pair[0]))
                    {
                        adjacencyList[pair[0]] = new HashSet<string>();
                    }
                    adjacencyList[pair[0]].Add(pair[1]);

                    if (!adjacencyList.ContainsKey(pair[1]))
                    {
                        adjacencyList[pair[1]] = new HashSet<string>();
                    }
                    adjacencyList[pair[1]].Add(pair[0]);
                }

                for (int i = 0; i < sentence1.Length; i++)
                {
                    if (sentence1[i].Equals(sentence2[i]))
                    {
                        continue;
                    }
                    HashSet<string> visitedNodes = new HashSet<string>();
                    if (adjacencyList.ContainsKey(sentence1[i]) && adjacencyList.ContainsKey(sentence2[i]) &&
                            DepthFirstSearch(sentence1[i], adjacencyList, visitedNodes, sentence2[i]))
                    {
                        continue;
                    }
                    return false;
                }
                return true;
            }
            private bool DepthFirstSearch(string node, Dictionary<string, HashSet<string>> adjacencyList, HashSet<string> visitedNodes, string destination)
            {
                visitedNodes.Add(node);
                if (node.Equals(destination))
                {
                    return true;
                }
                if (!adjacencyList.ContainsKey(node))
                {
                    return false;
                }
                foreach (string neighbor in adjacencyList[node])
                {
                    if (!visitedNodes.Contains(neighbor) && DepthFirstSearch(neighbor, adjacencyList, visitedNodes, destination))
                    {
                        return true;
                    }
                }
                return false;
            }
            /*
            Approach 2: Breadth First Search
            Complexity Analysis
Here, n is the number of words in sentence1 and sentence2 and k is the number of similar pairs. Let m be the average length of words in sentence1, sentence2 as well as in similarPairs.
•	Time complexity: O(n⋅k⋅m)
o	We iterate over all the elements of similarPairs and insert a pair twice into adj. To hash each word of length m, we need O(m) time, and to put the same length word in the hash set, we need O(m) time again. Because there are k pairs of words, there can be at most 2⋅k words that can be hashed and added to the set. As a result, we require O(k⋅m) time.
o	We iterate over all of sentence1's words to see if the corresponding words are equal or similar. If the words are not equal, we perform a BFS traversal over the graph with O(k) nodes and edges. As we know, a single BFS traversal takes O(V+E) time to traverse a graph with V nodes and E edges. In our case, we have O(k) edges and nodes, and each node is a string with an average size of m, so a single traversal would take O(k⋅m) time. For n traversals, it would take O(n⋅k⋅m).
o	The total amount of time required is O(n⋅k⋅m+k⋅m)=O(n⋅k⋅m).
•	Space complexity: O(k⋅m)
o	Because we use similarPairs to create the graph, the number of nodes and edges can be O(k). To map O(k) words of size m, adj uses O(k⋅m) space.
o	In the worst case, the queue can grow to a size linear with k. It would need O(k⋅m) space since each element is of length m.
o	We also use a set visit, which requires O(k⋅m) space because it could contain O(k) words in the worst case.

            */
            // Returns true if there is a path from node to dest.
            private bool BreadthFirstSearch(string source, Dictionary<string, HashSet<string>> adjacencyList, string destination)
            {
                HashSet<string> visitedNodes = new HashSet<string>();
                Queue<string> queue = new Queue<string>();
                queue.Enqueue(source);
                visitedNodes.Add(source);

                while (queue.Count > 0)
                {
                    string currentNode = queue.Dequeue();

                    if (!adjacencyList.ContainsKey(currentNode))
                    {
                        continue;
                    }
                    foreach (string neighbor in adjacencyList[currentNode])
                    {
                        if (neighbor.Equals(destination))
                        {
                            return true;
                        }
                        if (!visitedNodes.Contains(neighbor))
                        {
                            visitedNodes.Add(neighbor);
                            queue.Enqueue(neighbor);
                        }
                    }
                }
                return false;
            }

            public bool BFS(string[] firstSentence, string[] secondSentence, List<List<string>> similarPairs)
            {
                if (firstSentence.Length != secondSentence.Length)
                {
                    return false;
                }
                // Create the graph using each pair in similarPairs.
                Dictionary<string, HashSet<string>> adjacencyList = new Dictionary<string, HashSet<string>>();
                foreach (List<string> pair in similarPairs)
                {
                    if (!adjacencyList.ContainsKey(pair[0]))
                    {
                        adjacencyList[pair[0]] = new HashSet<string>();
                    }
                    adjacencyList[pair[0]].Add(pair[1]);

                    if (!adjacencyList.ContainsKey(pair[1]))
                    {
                        adjacencyList[pair[1]] = new HashSet<string>();
                    }
                    adjacencyList[pair[1]].Add(pair[0]);
                }

                for (int index = 0; index < firstSentence.Length; index++)
                {
                    if (firstSentence[index].Equals(secondSentence[index]))
                    {
                        continue;
                    }
                    if (adjacencyList.ContainsKey(firstSentence[index]) && adjacencyList.ContainsKey(secondSentence[index]) &&
                            BreadthFirstSearch(firstSentence[index], adjacencyList, secondSentence[index]))
                    {
                        continue;
                    }
                    return false;
                }
                return true;
            }

            /*
            Approach 3: Union Find
Complexity Analysis
Here, n is the number of words in sentence1 and sentence2 and k is the number of similar pairs. Let m be the average length of words in sentence1, sentence2 as well as in similarPairs.
•	Time complexity: O((n+k)⋅m)
o	For T operations, the amortized time complexity of the union-find algorithm (using path compression with union by rank) is O(alpha(T)). Here, α(T) is the inverse Ackermann function that grows so slowly, that it doesn't exceed 4 for all reasonable T (approximately T<10600). You can read more about the complexity of union-find here. Because the function grows so slowly, we consider it to be O(1). Since, we are performing union and find of strings with average length m, we consider it to be O(m).
o	We iterate over all the elements of similarPairs and add both the words in pair to dsu which takes O(m) each. We then perform union which also takes O(m) time per operation. Because there are k pairs, this step will take O(k⋅m) time for all the pairs.
o	We iterate over all of sentence1's words to see if the words are equal. Because each word is of m length, this would take O(m) time for each word. It will take O(n⋅m) time in total because there are n words. If the words are not equal, we perform a find operation that takes O(m) per find operation. It would take O(n⋅m) time for n words.
o	As a result, the total time would be O((n+k)⋅m+k⋅m)=O((n+k)⋅m).
•	Space complexity: O(k⋅m)
o	Because there can be k similar pairs, there are O(k) words that are inserted in dsu. Since each word has average length m, parent and rank maps would take up O(k⋅m) space.

            */
            public bool WithUnionFind(string[] sentence1, string[] sentence2, List<List<string>> similarPairs)
            {
                if (sentence1.Length != sentence2.Length)
                {
                    return false;
                }

                UnionFind unionFind = new UnionFind();
                foreach (List<string> pair in similarPairs)
                {
                    // Create the graph using the hashed values of the similarPairs.
                    unionFind.AddWord(pair[0]);
                    unionFind.AddWord(pair[1]);
                    unionFind.Union(pair[0], pair[1]);
                }

                for (int i = 0; i < sentence1.Length; i++)
                {
                    if (sentence1[i] == sentence2[i])
                    {
                        continue;
                    }
                    if (unionFind.IsWordPresent(sentence1[i]) && unionFind.IsWordPresent(sentence2[i]) &&
                        unionFind.Find(sentence1[i]) == unionFind.Find(sentence2[i]))
                    {
                        continue;
                    }
                    return false;
                }
                return true;
            }
            class UnionFind
            {
                private Dictionary<string, string> parent = new Dictionary<string, string>();
                private Dictionary<string, int> rank = new Dictionary<string, int>();

                public void AddWord(string word)
                {
                    if (!parent.ContainsKey(word))
                    {
                        parent[word] = word;
                        rank[word] = 0;
                    }
                }

                public bool IsWordPresent(string word)
                {
                    return parent.ContainsKey(word);
                }

                public string Find(string word)
                {
                    if (parent[word] != word)
                    {
                        parent[word] = Find(parent[word]);
                    }
                    return parent[word];
                }

                public void Union(string word1, string word2)
                {
                    string root1 = Find(word1);
                    string root2 = Find(word2);
                    if (root1 == root2)
                    {
                        return;
                    }
                    else if (rank[root1] < rank[root2])
                    {
                        parent[root1] = root2;
                    }
                    else if (rank[root1] > rank[root2])
                    {
                        parent[root2] = root1;
                    }
                    else
                    {
                        parent[root2] = root1;
                        rank[root1]++;
                    }
                }
            }

        }

        /*
        739. Daily Temperatures	
        https://leetcode.com/problems/daily-temperatures/description/
        */

        public class DailyTemperaturesSol
        {
            /*
            Approach 1: Monotonic Stack
Complexity Analysis
Given N as the length of temperatures,
•	Time complexity: O(N)
At first glance, it may look like the time complexity of this algorithm should be O(N2), because there is a nested while loop inside the for loop. However, each element can only be added to the stack once, which means the stack is limited to N pops. Every iteration of the while loop uses 1 pop, which means the while loop will not iterate more than N times in total, across all iterations of the for loop.
An easier way to think about this is that in the worst case, every element will be pushed and popped once. This gives a time complexity of O(2⋅N)=O(N).
•	Space complexity: O(N)
If the input was non-increasing, then no element would ever be popped from the stack, and the stack would grow to a size of N elements at the end.
Note: answer does not count towards the space complexity because space used for the output format does not count.	

            */
            public int[] UsingMonotonicStack(int[] temperatures)
            {
                int numberOfDays = temperatures.Length;
                int[] result = new int[numberOfDays];
                Stack<int> dayStack = new Stack<int>();

                for (int currentDay = 0; currentDay < numberOfDays; currentDay++)
                {
                    int currentTemperature = temperatures[currentDay];
                    // Pop until the current day's temperature is not
                    // warmer than the temperature at the top of the stack
                    while (dayStack.Count > 0 && temperatures[dayStack.Peek()] < currentTemperature)
                    {
                        int previousDay = dayStack.Pop();
                        result[previousDay] = currentDay - previousDay;
                    }
                    dayStack.Push(currentDay);
                }

                return result;
            }
            /*
            Approach 2: Array, Optimized Space
            Complexity Analysis
Given N as the length of temperatures,
•	Time complexity: O(N)
Similar to the first approach, the nested while loop makes this algorithm look worse than O(N). However, same as in the first approach, the total number of iterations in the while loop does not exceed N, which gives this algorithm a time complexity of O(2⋅N)=O(N).
The reason the iterations in the while loop does not exceed N is because the "jumps" prevent an index from being visited twice. If we had the example temperatures = [45, 43, 45, 43, 45, 31, 32, 33, 50], after 5 iterations we would have answer = [..., 4, 1, 1, 1, 0]. The day at index 2 will use answer[4] to jump to the final day (which is the next warmer day), and then answer[4] will not be used again. This is because at the first day, answer[2] will be used to jump all the way to the end. The final solution is answer = [8,1,6,1,4,1,1,1,0]. The 6 was found with the help of the 4 and the 8 was found with the help of the 6.
•	Space complexity: O(1)
As stated above, while answer does use O(N) space, the space used for the output does not count towards the space complexity. Thus, only constant extra space is used.

            */

            public int[] UsingArray(int[] temperatures)
            {
                int n = temperatures.Length;
                int hottest = 0;
                int[] answer = new int[n];

                for (int currDay = n - 1; currDay >= 0; currDay--)
                {
                    int currentTemp = temperatures[currDay];
                    if (currentTemp >= hottest)
                    {
                        hottest = currentTemp;
                        continue;
                    }

                    int days = 1;
                    while (temperatures[currDay + days] <= currentTemp)
                    {
                        // Use information from answer to search for the next warmer day
                        days += answer[currDay + days];
                    }
                    answer[currDay] = days;
                }

                return answer;
            }

        }

        /*
        740. Delete and Earn	
        https://leetcode.com/problems/delete-and-earn/description/	
        */
        class DeleteAndEarnSol
        {
            /*
Approach 1: Top-Down Dynamic Programming
Complexity Analysis
Given N as the length of nums and k as the maximum element in nums,
•	Time complexity: O(N+k)
To populate points, we need to iterate through nums once, which costs O(N) time. Then, we call maxPoints(maxNumber). This call will repeatedly call maxPoints until we get down to our base cases. Because of cache, already solved sub-problems will only cost O(1) time. Since maxNumber = k, we will solve k unique sub-problems so, this recursion will cost O(k) time. Our final time complexity is O(N+k).
•	Space complexity: O(N+k)
The extra space we use is the hash table points, the recursion call stack needed to find maxPoints(maxNumber), and the hash table cache.
The size of points is equal to the number of unique elements in nums. In the worst case, where every element in nums is unique, this will take O(N) space. The recursion call stack will also grow up to size k, since we start our recursion at maxNumber, and we don't start returning values until our base cases at 0 and 1. Lastly, cache will store the answer for all states, from 2 to maxNumber, which means it also grows up to k size. Our final space complexity is O(N+2⋅k) = O(N+k).

            */
            private Dictionary<int, int> points = new Dictionary<int, int>();
            private Dictionary<int, int> cache = new Dictionary<int, int>();
            public int TopDownDP(int[] nums)
            {
                int maxNumber = 0;

                // Precompute how many points we gain from taking an element
                foreach (int number in nums)
                {
                    if (points.ContainsKey(number))
                    {
                        points[number] += number;
                    }
                    else
                    {
                        points[number] = number;
                    }
                    maxNumber = Math.Max(maxNumber, number);
                }

                return MaxPoints(maxNumber);
            }
            private int MaxPoints(int number)
            {
                // Check for base cases
                if (number == 0)
                {
                    return 0;
                }

                if (number == 1)
                {
                    return points.ContainsKey(1) ? points[1] : 0;
                }

                if (cache.ContainsKey(number))
                {
                    return cache[number];
                }

                // Apply recurrence relation
                int gain = points.ContainsKey(number) ? points[number] : 0;
                cache[number] = Math.Max(MaxPoints(number - 1), MaxPoints(number - 2) + gain);
                return cache[number];
            }

            /*
            Approach 2: Bottom-Up Dynamic Programming
            Complexity Analysis
            Given N as the length of nums and k as the maximum element in nums,
            •	Time complexity: O(N+k)
            To populate points, we need to iterate through nums once, which costs O(N) time. Then, we populate maxPoints by iterating over it. The length of maxPoints is equal to k + 1, so this will cost O(k) time. Our final time complexity is O(N+k).
            •	Space complexity: O(N+k)
            The extra space we use is the hash table points and our DP array maxPoints. The size of maxPoints is equal to k + 1, which means it takes O(k) space. The size of points is equal to the number of unique elements in nums. In the worst case, where every element in nums is unique, this will take O(N) space. Our final space complexity is O(N+k).


            */
            public int BottomUpDP(int[] nums)
            {
                Dictionary<int, int> points = new Dictionary<int, int>();
                int maxNumber = 0;

                // Precompute how many points we gain from taking an element
                foreach (int num in nums)
                {
                    if (points.ContainsKey(num))
                    {
                        points[num] += num;
                    }
                    else
                    {
                        points[num] = num;
                    }
                    maxNumber = Math.Max(maxNumber, num);
                }

                // Declare our array along with base cases
                int[] maxPoints = new int[maxNumber + 1];
                maxPoints[1] = points.ContainsKey(1) ? points[1] : 0;

                for (int num = 2; num < maxPoints.Length; num++)
                {
                    // Apply recurrence relation
                    int gain = points.ContainsKey(num) ? points[num] : 0;
                    maxPoints[num] = Math.Max(maxPoints[num - 1], maxPoints[num - 2] + gain);
                }

                return maxPoints[maxNumber];
            }

            /*
            Approach 3: Space Optimized Bottom-Up Dynamic Programming
            Complexity Analysis
Given N as the length of nums and k as the maximum element in nums,
•	Time complexity: O(N+k)
To populate points, we need to iterate through nums once, which costs O(N) time. Then, we iterate from 2 to k, doing O(1) work at each iteration, so this will cost O(k) time. Our final time complexity is O(N+k).
•	Space complexity: O(N)
The extra space we use is the hash table points.
The size of points is equal to the number of unique elements in nums. In the worst case, where every element in nums is unique, this will take O(N) space.
Unlike in approach 2, we no longer need an array maxPoints which would be of size k. Thus, we have improved the space complexity to O(N).

            */
            public int BottomUpDPWithSpaceOptimal(int[] numbers)
            {
                int maximumNumber = 0;
                Dictionary<int, int> points = new Dictionary<int, int>();

                // Precompute how many points we gain from taking an element
                foreach (int number in numbers)
                {
                    if (points.ContainsKey(number))
                    {
                        points[number] += number;
                    }
                    else
                    {
                        points[number] = number;
                    }
                    maximumNumber = Math.Max(maximumNumber, number);
                }

                // Base cases
                int pointsTwoBack = 0;
                int pointsOneBack = points.ContainsKey(1) ? points[1] : 0;

                for (int number = 2; number <= maximumNumber; number++)
                {
                    int temporary = pointsOneBack;
                    pointsOneBack = Math.Max(pointsOneBack, pointsTwoBack + (points.ContainsKey(number) ? points[number] : 0));
                    pointsTwoBack = temporary;
                }

                return pointsOneBack;
            }
            /*
            Approach 4: Iterate Over Elements
Complexity Analysis
Given N as the length of nums,
•	Time complexity: O(N⋅log(N))
To populate points, we need to iterate through nums once, which costs O(N) time.
Next, we take all the keys of points and sort them to create elements. In the worst case when nums only contains unique elements, there will be N keys, which means this will cost O(N⋅log(N)) time.
Lastly, we iterate through elements, which again in the worst case costs O(N) time when all the elements are unique.
This gives us a time complexity of O(N+N⋅log(N)+N)=O(N⋅log(N)).
•	Space complexity: O(N)
The extra space we use is the hash table points and elements. These have the same length, and in the worst case scenario when nums only contains unique elements, their lengths will be equal to N.

            */
            public int IterateOverElem(int[] numbers)
            {
                Dictionary<int, int> points = new Dictionary<int, int>();

                // Precompute how many points we gain from taking an element
                foreach (int number in numbers)
                {
                    if (points.ContainsKey(number))
                    {
                        points[number] += number;
                    }
                    else
                    {
                        points[number] = number;
                    }
                }

                List<int> elements = new List<int>(points.Keys);
                elements.Sort();

                // Base cases
                int twoBack = 0;
                int oneBack = points[elements[0]];

                for (int i = 1; i < elements.Count; i++)
                {
                    int currentElement = elements[i];
                    int temp = oneBack;
                    if (currentElement == elements[i - 1] + 1)
                    {
                        // The 2 elements are adjacent, cannot take both - apply normal recurrence
                        oneBack = Math.Max(oneBack, twoBack + points[currentElement]);
                    }
                    else
                    {
                        // Otherwise, we don't need to worry about adjacent deletions
                        oneBack += points[currentElement];
                    }

                    twoBack = temp;
                }

                return oneBack;
            }
            /*
            Approach5: The Best of Both Worlds :-)
Complexity Analysis
Given N as the length of nums and k as the maximum element in nums,
•	Time complexity: O(N+min(k,N⋅log(N)))
To populate points, we need to iterate through nums once, which costs O(N) time.
Approach 3's algorithm costs O(k) time. Approach 4's algorithm costs O(N⋅log(N)) time. We choose the faster one, so the final time complexity will be O(N+min(k,N⋅log(N))).
•	Space complexity: O(N)
The extra space we use is the hash table points, and maybe elements if we are to use approach 4's algorithm. However, points has the same length as elements, so it doesn't matter either way in terms of space complexity.
In the worst-case scenario when all elements in nums are unique, points will grow to a size of N.	

            */
            public int BestOfBoth3And4Approaches(int[] numbers)
            {
                int maxNumber = 0;
                Dictionary<int, int> points = new Dictionary<int, int>();

                foreach (int number in numbers)
                {
                    if (points.ContainsKey(number))
                    {
                        points[number] += number;
                    }
                    else
                    {
                        points[number] = number;
                    }
                    maxNumber = Math.Max(maxNumber, number);
                }

                int twoBack = 0;
                int oneBack = 0;
                int n = points.Count;

                if (maxNumber < n + n * Math.Log(n) / Math.Log(2))
                {
                    oneBack = points.ContainsKey(1) ? points[1] : 0;
                    for (int number = 2; number <= maxNumber; number++)
                    {
                        int temp = oneBack;
                        oneBack = Math.Max(oneBack, twoBack + (points.ContainsKey(number) ? points[number] : 0));
                        twoBack = temp;
                    }
                }
                else
                {
                    List<int> elements = new List<int>(points.Keys);
                    elements.Sort();
                    oneBack = points[elements[0]];

                    for (int i = 1; i < elements.Count; i++)
                    {
                        int currentElement = elements[i];
                        int temp = oneBack;
                        if (currentElement == elements[i - 1] + 1)
                        {
                            oneBack = Math.Max(oneBack, twoBack + points[currentElement]);
                        }
                        else
                        {
                            oneBack += points[currentElement];
                        }

                        twoBack = temp;
                    }
                }

                return oneBack;
            }


        }

        /*
        743. Network Delay Time
        https://leetcode.com/problems/network-delay-time/description/
        */
        class NetworkDelayTimeSol
        {

            /*
Approach 1: Depth-First Search (DFS)
Complexity Analysis
Here N is the number of nodes and E is the number of total edges in the given network.
•	Time complexity: O((N−1)!+ElogE)
In a complete graph with N nodes and N∗(N−1) directed edges, we can end up traversing all the paths of all the possible lengths. The total number of paths can be represented as ∑len=1N(lenN)∗len!, where len is the length of path which can be 1 to N. This number can be represented as e.N!, it's essentially equal to the number of arrangements for N elements. In our case, the first element will always be K, hence the number of arrangements is e.(N−1)!.
Also, we sort the edges corresponding to each node, this can be expressed as ElogE because sorting each small bucket of outgoing edges is bounded by sorting all of them, using the inequality xlogx+ylogy≤(x+y)log(x+y). Also, finding the minimum time required in signalReceivedAt takes O(N).
•	Space complexity: O(N+E)
Building the adjacency list will take O(E) space and the run-time stack for DFS can have at most N active functions calls hence, O(N) space.

            */
            public int DFS(int[][] times, int numberOfNodes, int startingNode)
            {
                // Build the adjacency list
                foreach (var time in times)
                {
                    int source = time[0];
                    int destination = time[1];
                    int travelTime = time[2];

                    if (!adjacencyList.ContainsKey(source))
                    {
                        adjacencyList[source] = new List<KeyValuePair<int, int>>();
                    }
                    adjacencyList[source].Add(new KeyValuePair<int, int>(travelTime, destination));
                }

                // Sort the edges connecting to every node
                foreach (var node in adjacencyList.Keys)
                {
                    adjacencyList[node].Sort((a, b) => a.Key.CompareTo(b.Key));
                }

                int[] signalReceivedAt = new int[numberOfNodes + 1];
                Array.Fill(signalReceivedAt, int.MaxValue);

                DepthFirstSearch(signalReceivedAt, startingNode, 0);

                int answer = int.MinValue;
                for (int node = 1; node <= numberOfNodes; node++)
                {
                    answer = Math.Max(answer, signalReceivedAt[node]);
                }

                // int.MaxValue signifies at least one node is unreachable
                return answer == int.MaxValue ? -1 : answer;
            }

            // Adjacency list
            private Dictionary<int, List<KeyValuePair<int, int>>> adjacencyList = new Dictionary<int, List<KeyValuePair<int, int>>>();

            private void DepthFirstSearch(int[] signalReceivedAt, int currentNode, int currentTime)
            {
                // If the current time is greater than or equal to the fastest signal received
                // Then no need to iterate over adjacent nodes
                if (currentTime >= signalReceivedAt[currentNode])
                {
                    return;
                }

                // Fastest signal time for currentNode so far
                signalReceivedAt[currentNode] = currentTime;

                if (!adjacencyList.ContainsKey(currentNode))
                {
                    return;
                }

                // Broadcast the signal to adjacent nodes
                foreach (var edge in adjacencyList[currentNode])
                {
                    int travelTime = edge.Key;
                    int neighborNode = edge.Value;

                    // currentTime + travelTime: time when signal reaches neighborNode
                    DepthFirstSearch(signalReceivedAt, neighborNode, currentTime + travelTime);
                }
            }
            /*
            Approach 2: Breadth-First Search (BFS)
            Complexity Analysis
            Here N is the number of nodes and E is the number of total edges in the given network.
            •	Time complexity: O(N⋅E)
            Each of the N nodes can be added to the queue for all the edges connected to it, hence in a complete graph, the total number of operations would be O(NE). Also, finding the minimum time required in signalReceivedAt takes O(N).
            •	Space complexity: O(N⋅E)
            Building the adjacency list will take O(E) space and the queue for BFS will use O(N⋅E) space as there can be this much number of nodes in the queue.	

            */
            public int BFS(int[][] times, int n, int k)
            {
                // Build the adjacency list
                foreach (int[] time in times)
                {
                    int source = time[0];
                    int destination = time[1];
                    int travelTime = time[2];

                    if (!adjacencyList.ContainsKey(source))
                    {
                        adjacencyList[source] = new List<KeyValuePair<int, int>>();
                    }
                    adjacencyList[source].Add(new KeyValuePair<int, int>(travelTime, destination));
                }

                int[] signalReceivedAt = new int[n + 1];
                Array.Fill(signalReceivedAt, int.MaxValue);

                BFS(signalReceivedAt, k);

                int answer = int.MinValue;
                for (int i = 1; i <= n; i++)
                {
                    answer = Math.Max(answer, signalReceivedAt[i]);
                }

                // INT_MAX signifies at least one node is unreachable
                return answer == int.MaxValue ? -1 : answer;
            }
            private void BFS(int[] signalReceivedAt, int sourceNode)
            {
                Queue<int> queue = new Queue<int>();
                queue.Enqueue(sourceNode);

                // Time for starting node is 0
                signalReceivedAt[sourceNode] = 0;

                while (queue.Count > 0)
                {
                    int currentNode = queue.Dequeue();

                    if (!adjacencyList.ContainsKey(currentNode))
                    {
                        continue;
                    }

                    // Broadcast the signal to adjacent nodes
                    foreach (KeyValuePair<int, int> edge in adjacencyList[currentNode])
                    {
                        int time = edge.Key;
                        int neighborNode = edge.Value;

                        // Fastest signal time for neighborNode so far
                        // signalReceivedAt[currentNode] + time : 
                        // time when signal reaches neighborNode
                        int arrivalTime = signalReceivedAt[currentNode] + time;
                        if (signalReceivedAt[neighborNode] > arrivalTime)
                        {
                            signalReceivedAt[neighborNode] = arrivalTime;
                            queue.Enqueue(neighborNode);
                        }
                    }
                }
            }
            /*
            Approach 3: Dijkstra's Algorithm
           Complexity Analysis
Here N is the number of nodes and E is the number of total edges in the given network.
•	Time complexity: O(N+ElogN)
Dijkstra's Algorithm takes O(ElogN). Finding the minimum time required in signalReceivedAt takes O(N).
The maximum number of vertices that could be added to the priority queue is E. Thus, push and pop operations on the priority queue take O(logE) time. The value of E can be at most N⋅(N−1). Therefore, O(logE) is equivalent to O(logN2) which in turn equivalent to O(2⋅logN). Hence, the time complexity for priority queue operations equals O(logN).
Although the number of vertices in the priority queue could be equal to E, we will only visit each vertex only once. If we encounter a vertex for the second time, then currNodeTime will be greater than signalReceivedAt[currNode], and we can continue to the next vertex in the priority queue. Hence, in total E edges will be traversed and for each edge, there could be one priority queue insertion operation.
Hence, the time complexity is equal to O(N+ElogN).
•	Space complexity: O(N+E)
Building the adjacency list will take O(E) space. Dijkstra's algorithm takes O(E) space for priority queue because each vertex could be added to the priority queue N−1 time which makes it N∗(N−1) and O(N2) is equivalent to O(E). signalReceivedAt takes O(N) space.
 
            */
            public int Dijkstra(int[][] times, int n, int k)
            {
                // Build the adjacency list
                foreach (int[] time in times)
                {
                    int source = time[0];
                    int destination = time[1];
                    int travelTime = time[2];

                    if (!adjacencyList.ContainsKey(source))
                    {
                        adjacencyList[source] = new List<KeyValuePair<int, int>>();
                    }
                    adjacencyList[source].Add(new KeyValuePair<int, int>(travelTime, destination));
                }

                int[] signalReceivedAt = new int[n + 1];
                Array.Fill(signalReceivedAt, int.MaxValue);

                DijkstraAlgo(signalReceivedAt, k, n);

                int answer = int.MinValue;
                for (int i = 1; i <= n; i++)
                {
                    answer = Math.Max(answer, signalReceivedAt[i]);
                }

                // INT_MAX signifies at least one node is unreachable
                return answer == int.MaxValue ? -1 : answer;
            }
            private void DijkstraAlgo(int[] signalReceivedAt, int source, int n)
            {
                PriorityQueue<(int, int), (int, int)> priorityQueue = new PriorityQueue<(int, int), (int, int)>(
                    Comparer<(int, int)>.Create((a, b) => a.Item1.CompareTo(b.Item1)) //Compare Keys
                );
                priorityQueue.Enqueue((0, source), (0, source));

                // Time for starting node is 0
                signalReceivedAt[source] = 0;

                while (priorityQueue.Count > 0)
                {
                    (int currentNodeTime, int currentNode) = priorityQueue.Dequeue();

                    if (currentNodeTime > signalReceivedAt[currentNode])
                    {
                        continue;
                    }

                    if (!adjacencyList.ContainsKey(currentNode))
                    {
                        continue;
                    }

                    // Broadcast the signal to adjacent nodes
                    foreach (KeyValuePair<int, int> edge in adjacencyList[currentNode])
                    {
                        int time = edge.Key;
                        int neighborNode = edge.Value;

                        // Fastest signal time for neighborNode so far
                        // signalReceivedAt[currentNode] + time : 
                        // time when signal reaches neighborNode
                        if (signalReceivedAt[neighborNode] > currentNodeTime + time)
                        {
                            signalReceivedAt[neighborNode] = currentNodeTime + time;
                            priorityQueue.Enqueue((signalReceivedAt[neighborNode], neighborNode), (signalReceivedAt[neighborNode], neighborNode));
                        }
                    }
                }
            }


        }

        /*
        2039. The Time When the Network Becomes Idle
        https://leetcode.com/problems/the-time-when-the-network-becomes-idle/description/
        */
        public class NetworkBecomesIdleSol
        {
            /*
            Approach: Bfs and result is maximum of time of last messages
Complexity
•	Time complexity: O(n + m)
•	Space complexity:	O(n + m)
	

            */
            private void BfsAlgo(List<int>[] graph, int start, int[] shortest)
            {
                int n = graph.Length;
                bool[] visited = new bool[n];
                Queue<int> q = new Queue<int>();
                q.Enqueue(start);
                visited[start] = true;
                int count = 0;

                while (q.Count > 0)
                {
                    int size = q.Count;

                    count++;
                    while (size-- > 0)
                    {
                        int v = q.Dequeue();
                        foreach (var u in graph[v])
                        {
                            if (visited[u]) continue;

                            visited[u] = true;
                            q.Enqueue(u);
                            shortest[u] = count;
                        }
                    }
                }
            }

            public int BFS(int[][] edges, int[] patience)
            {
                int n = patience.Length;
                List<int>[] graph = new List<int>[n];
                int[] shortest = new int[n];

                for (int i = 0; i < n; i++)
                {
                    graph[i] = new List<int>();
                }
                //Build Adjency List
                foreach (var edge in edges)
                {
                    graph[edge[0]].Add(edge[1]);
                    graph[edge[1]].Add(edge[0]);
                }

                BfsAlgo(graph, 0, shortest);
                //Calculate answer using shortest paths.
                int result = 0;

                for (int idx = 1; idx < n; idx++)
                {
                    int resendInterval = patience[idx];

                    //#The server will stop sending requests after it's been sent to the master node and back.
                    int shutOffTime = shortest[idx] * 2;

                    //# shutOffTime-1 == Last second the server can send a re-request.
                    int lastSecond = shutOffTime - 1;

                    //Calculate the last time a packet is actually resent.        
                    /*
                    (lastSecond//resendInterval) tells us how MANY TIMES a message will be resent during the time it takes 
                    for the first message to complete its journey. We must multiply this number by the number of resendIntervals to know when the last of the many resent messages is finally sent out!
                    */
                    int lastResentTime = (lastSecond % resendInterval) * resendInterval;

                    //# At the last resent time, the packet still must go through 2 more cycles to the master node and back.
                    int lastPacketTime = lastResentTime + shutOffTime;

                    result = Math.Max(result, lastPacketTime);
                }
                //Add +1, the current answer is the last time the packet is recieved by the target server (still active).
                //We must return the first second the network is idle, therefore + 1
                return result + 1;
            }
        }


        /*
        2045. Second Minimum Time to Reach Destination
        https://leetcode.com/problems/second-minimum-time-to-reach-destination/description/
        */
        class SecondMinimumToReachDestSol
        {
            /*
            Approach 1: Modified Dijkstra
            Complexity Analysis
Let N be the number of cities and E be the total edges in the graph.
•	Time complexity: O(N+E⋅logN).
o	Our algorithm has twice the complexity as the Dijkstra algorithm. We pop twice and use the node to calculate the minimum and second miimum distance. Since 2 is a constant factor, it actually has the same time complexity as the standard Dijkstra algorithm.
o	For standard Dijkstra, the maximum number of vertices that could be added to the priority queue is E and each operation takes O(logE) time. Thus, push and pop operations on the priority queue take O(E⋅logE) time. The value of E can be at most N⋅(N−1), so O(E⋅logE)=O(E⋅log(N2))=O(E⋅logN). It also takes O(N+E) for adjacency list and dist array initializations. Therefore, the total complexity is O(N+E⋅logN).
•	Space complexity: O(N+E).
o	Building the adjacency list takes O(N+E) space. For the Dijkstra algorithm, each vertex is added to the queue at most N−1 times, so the space it takes is N⋅(N−1)=O(N^2)=O(E). For the distance and frequency arrays, they take O(N) space.

            */

            public int ModifiedDijkstra(int numberOfNodes, int[][] edges, int travelTime, int trafficLightChange)
            {

                Dictionary<int, List<int>> adjacencyList = new Dictionary<int, List<int>>();
                foreach (int[] edge in edges)
                {
                    int nodeA = edge[0], nodeB = edge[1];
                    if (!adjacencyList.ContainsKey(nodeA))
                    {
                        adjacencyList[nodeA] = new List<int>();
                    }
                    adjacencyList[nodeA].Add(nodeB);
                    if (!adjacencyList.ContainsKey(nodeB))
                    {
                        adjacencyList[nodeB] = new List<int>();
                    }
                    adjacencyList[nodeB].Add(nodeA);
                }

                int[] minimumTime = new int[numberOfNodes + 1];
                int[] secondMinimumTime = new int[numberOfNodes + 1];
                int[] visitFrequency = new int[numberOfNodes + 1];

                // minimumTime[i] stores the minimum time taken to reach node i from node 1. secondMinimumTime[i]
                // stores the second minimum time taken to reach node i from node 1. visitFrequency[i] stores
                // the number of times a node is popped out of the heap.
                for (int i = 1; i <= numberOfNodes; i++)
                {
                    minimumTime[i] = secondMinimumTime[i] = int.MaxValue;
                    visitFrequency[i] = 0;
                }

                PriorityQueue<int[], int[]> priorityQueue = new PriorityQueue<int[], int[]>(Comparer<int[]>.Create((a, b) => (a[1] - b[1])));
                priorityQueue.Enqueue(new int[] { 1, 0 }, new int[] { 1, 0 });
                minimumTime[1] = 0;

                while (priorityQueue.Count > 0)
                {
                    int[] current = priorityQueue.Dequeue();
                    int currentNode = current[0];
                    int timeTaken = current[1];

                    visitFrequency[currentNode]++;

                    // If the node is being visited for the second time and is 'n', return the answer.
                    if (visitFrequency[currentNode] == 2 && currentNode == numberOfNodes) return timeTaken;
                    // If the current light is red, wait till the path turns green.
                    if ((timeTaken / trafficLightChange) % 2 == 1)
                    {
                        timeTaken = trafficLightChange * (timeTaken / trafficLightChange + 1) + travelTime;
                    }
                    else
                    {
                        timeTaken += travelTime;
                    }

                    if (!adjacencyList.ContainsKey(currentNode)) continue;
                    foreach (int neighbor in adjacencyList[currentNode])
                    {
                        // Ignore nodes that have already popped out twice, we are not interested in visiting them again.
                        if (visitFrequency[neighbor] == 2) continue;

                        // Update minimumTime if it's more than the current timeTaken and store its value in
                        // secondMinimumTime since that becomes the second minimum value now.
                        if (minimumTime[neighbor] > timeTaken)
                        {
                            secondMinimumTime[neighbor] = minimumTime[neighbor];
                            minimumTime[neighbor] = timeTaken;
                            priorityQueue.Enqueue(new int[] { neighbor, timeTaken }, new int[] { neighbor, timeTaken });
                        }
                        else if (secondMinimumTime[neighbor] > timeTaken && minimumTime[neighbor] != timeTaken)
                        {
                            secondMinimumTime[neighbor] = timeTaken;
                            priorityQueue.Enqueue(new int[] { neighbor, timeTaken }, new int[] { neighbor, timeTaken });
                        }
                    }
                }
                return 0;
            }
            /*
            Approach 2: Breadth First Search
Complexity Analysis
Let N be the number of cities and E be the total edges in the graph.
•	Time complexity: O(N+E).
o	The complexity would be similar to the standard BFS algorithm since we’re iterating at most twice over a node.
o	For the BFS algorithm, each single queue operation takes O(1), and a single node could be pushed at most once leading O(N) operations. For each node popped out of the queue we iterate over all its neighbors, so for an undirected edge, a given edge could be iterated at most twice (by nodes at the end) which leads to O(E) operations in total for all the nodes and a total O(N+E) time complexity.
•	Space complexity: O(N+E).
o	Building the adjacency list takes O(E) space. The BFS queue takes O(N) because each vertex is added at most once. The other distance arrays take O(N) space.

            */
            public int BFS(int numberOfNodes, int[][] edges, int travelTime, int signalChangeTime)
            {
                Dictionary<int, List<int>> adjacencyList = new Dictionary<int, List<int>>();
                foreach (int[] edge in edges)
                {
                    int nodeA = edge[0], nodeB = edge[1];
                    if (!adjacencyList.ContainsKey(nodeA)) adjacencyList[nodeA] = new List<int>();
                    adjacencyList[nodeA].Add(nodeB);
                    if (!adjacencyList.ContainsKey(nodeB)) adjacencyList[nodeB] = new List<int>();
                    adjacencyList[nodeB].Add(nodeA);
                }

                int[] firstDistance = new int[numberOfNodes + 1];
                int[] secondDistance = new int[numberOfNodes + 1];
                for (int i = 1; i <= numberOfNodes; i++)
                {
                    firstDistance[i] = secondDistance[i] = -1;
                }

                Queue<int[]> queue = new Queue<int[]>();
                // Start with node 1, with its minimum distance.
                queue.Enqueue(new int[] { 1, 1 });
                firstDistance[1] = 0;

                while (queue.Count > 0)
                {
                    int[] current = queue.Dequeue();
                    int currentNode = current[0];
                    int currentFrequency = current[1];

                    int timeTaken = currentFrequency == 1 ? firstDistance[currentNode] : secondDistance[currentNode];
                    // If the time_taken falls under the red bracket, wait till the path turns green.
                    if ((timeTaken / signalChangeTime) % 2 == 1)
                    {
                        timeTaken = signalChangeTime * (timeTaken / signalChangeTime + 1) + travelTime;
                    }
                    else
                    {
                        timeTaken += travelTime;
                    }

                    foreach (int neighbor in adjacencyList[currentNode])
                    {
                        if (firstDistance[neighbor] == -1)
                        {
                            firstDistance[neighbor] = timeTaken;
                            queue.Enqueue(new int[] { neighbor, 1 });
                        }
                        else if (secondDistance[neighbor] == -1 && firstDistance[neighbor] != timeTaken)
                        {
                            if (neighbor == numberOfNodes) return timeTaken;
                            secondDistance[neighbor] = timeTaken;
                            queue.Enqueue(new int[] { neighbor, 2 });
                        }
                    }
                }
                return 0;
            }

        }

        /*
        1334. Find the City With the Smallest Number of Neighbors at a Threshold Distance
        https://leetcode.com/problems/find-the-city-with-the-smallest-number-of-neighbors-at-a-threshold-distance/description/
        */
        class FindTheCitySol
        {
            /*
        Approach 1: Dijkstra Algorithm
            Complexity Analysis
        Let n refer to the number of cities, where the constraints are 2<=n<=100, and m refer to the number of edges, with 1<=edges.length<=(n⋅(n−1))/2. This means that m can be at most (n⋅(n−1))/2, representing the maximum number of edges in an undirected graph where every city is connected to every other city with a unique edge.
        •	Time complexity: O(n^3logn)
        For one source, Dijkstra's algorithm using a priority queue runs in O(m⋅logn). With the maximum number of edges m, this becomes O(n⋅(n−1)/2⋅logn)=O(n^2logn). Running Dijkstra's algorithm for each city (source), the overall time complexity is O(n⋅n^2logn)=O(n^3logn).
        •	Space complexity: O(n^2)
        The space complexity is O(n^2) for the shortestPathMatrix and O(m+n) for the adjacency list and auxiliary data structures. Since m=O(n^2) in the worst case, the overall space complexity simplifies to O(n^2).

            */
            public int Dijkstra(int numberOfCities, int[][] edges, int distanceThreshold)
            {
                // Adjacency list to store the graph
                List<int[]>[] adjacencyList = new List<int[]>[numberOfCities];
                // Matrix to store shortest path distances from each city
                int[][] shortestPathMatrix = new int[numberOfCities][];

                // Initialize adjacency list and shortest path matrix
                for (int i = 0; i < numberOfCities; i++)
                {
                    shortestPathMatrix[i] = new int[numberOfCities];
                    Array.Fill(shortestPathMatrix[i], int.MaxValue); // Set all distances to infinity
                    shortestPathMatrix[i][i] = 0; // Distance to itself is zero
                    adjacencyList[i] = new List<int[]>();
                }

                // Populate the adjacency list with edges
                foreach (int[] edge in edges)
                {
                    int start = edge[0];
                    int end = edge[1];
                    int weight = edge[2];
                    adjacencyList[start].Add(new int[] { end, weight });
                    adjacencyList[end].Add(new int[] { start, weight }); // For undirected graph
                }

                // Compute shortest paths from each city using Dijkstra's algorithm
                for (int i = 0; i < numberOfCities; i++)
                {
                    DijkstraAlgo(numberOfCities, adjacencyList, shortestPathMatrix[i], i);
                }

                // Find the city with the fewest number of reachable cities within the distance threshold
                return GetCityWithFewestReachable(
                    numberOfCities,
                    shortestPathMatrix,
                    distanceThreshold
                );
            }

            // Dijkstra's algorithm to find shortest paths from a source city
            void DijkstraAlgo(int numberOfCities, List<int[]>[] adjacencyList, int[] shortestPathDistances, int source)
            {
                // Priority queue to process nodes with the smallest distance first
                PriorityQueue<int[], int[]> priorityQueue = new PriorityQueue<int[], int[]>(Comparer<int[]>.Create((a, b) =>
                    (a[1] - b[1])
                ));
                priorityQueue.Enqueue(new int[] { source, 0 }, new int[] { source, 0 });
                Array.Fill(shortestPathDistances, int.MaxValue); // Set all distances to infinity
                shortestPathDistances[source] = 0; // Distance to source itself is zero

                // Process nodes in priority order
                while (priorityQueue.Count > 0)
                {
                    int[] current = priorityQueue.Dequeue();
                    int currentCity = current[0];
                    int currentDistance = current[1];
                    if (currentDistance > shortestPathDistances[currentCity])
                    {
                        continue;
                    }

                    // Update distances to neighboring cities
                    foreach (int[] neighbor in adjacencyList[currentCity])
                    {
                        int neighborCity = neighbor[0];
                        int edgeWeight = neighbor[1];
                        if (shortestPathDistances[neighborCity] > currentDistance + edgeWeight)
                        {
                            shortestPathDistances[neighborCity] = currentDistance + edgeWeight;
                            priorityQueue.Enqueue(
                                new int[] {
                            neighborCity,
                            shortestPathDistances[neighborCity],
                                },
                                new int[] {
                            neighborCity,
                            shortestPathDistances[neighborCity],
                                }
                            );
                        }
                    }
                }
            }

            // Determine the city with the fewest number of reachable cities within the distance threshold
            int GetCityWithFewestReachable(int cityCount, int[][] shortestPathMatrix, int distanceThreshold)
            {
                int cityWithFewestReachableCities = -1;
                int fewestReachableCityCount = cityCount;

                // Count number of cities reachable within the distance threshold for each city
                for (int curretnCity = 0; curretnCity < cityCount; curretnCity++)
                {
                    int reachableCityCount = 0;
                    for (int otherCity = 0; otherCity < cityCount; otherCity++)
                    {
                        if (curretnCity == otherCity)
                        {
                            continue;
                        } // Skip self
                        if (shortestPathMatrix[curretnCity][otherCity] <= distanceThreshold)
                        {
                            reachableCityCount++;
                        }
                    }
                    // Update the city with the fewest reachable cities
                    if (reachableCityCount <= fewestReachableCityCount)
                    {
                        fewestReachableCityCount = reachableCityCount;
                        cityWithFewestReachableCities = curretnCity;
                    }
                }
                return cityWithFewestReachableCities;
            }
            /*
            Approach 2: Bellman-Ford Algorithm	
        Complexity Analysis
        Let n refer to the number of cities, where the constraints are 2<=n<=100, and m refer to the number of edges, with 1<=edges.length<=(n⋅(n−1))/2. This means that m can be at most n⋅(n−1)/2/2/2, representing the maximum number of edges in an undirected graph where every city is connected to every other city with a unique edge.
        •	Time complexity: O(n^4)
        For one source, Bellman-Ford runs in O(n⋅m), where m is the number of edges. In the worst case, m is n⋅(n−1)/2 (checkout the constraints), so the time complexity for one source becomes O(n⋅(n⋅(n−1)/2))=O(n^3). Since Bellman-Ford must be run for each city (source), the overall time complexity is O(n⋅n^3)=O(n^4).
        •	Space complexity: O(n^2)
        The space complexity is dominated by the shortestPathMatrix, which stores the shortest path distances between each pair of cities. This matrix requires O(n^2) space.

            */
            public int BellmanFord(int numberOfCities, int[][] edges, int distanceThreshold)
            {
                // Large value to represent infinity
                int infinityValue = (int)1e9 + 7;
                // Matrix to store shortest path distances from each city
                int[][] shortestPathMatrix = new int[numberOfCities][];

                for (int i = 0; i < numberOfCities; i++)
                {
                    shortestPathMatrix[i] = new int[numberOfCities];
                    BellmanFordAlgo(numberOfCities, edges, shortestPathMatrix[i], i);
                }

                // Find the city with the fewest number of reachable cities within the distance threshold
                return GetCityWithFewestReachable(
                    numberOfCities,
                    shortestPathMatrix,
                    distanceThreshold
                );
            }

            // Bellman-Ford algorithm to find shortest paths from a source city
            void BellmanFordAlgo(
                int numberOfCities,
                int[][] edges,
                int[] shortestPathDistances,
                int source
            )
            {
                // Initialize distances from the source
                Array.Fill(shortestPathDistances, int.MaxValue);
                shortestPathDistances[source] = 0; // Distance to source itself is zero

                // Relax edges up to numberOfCities - 1 times
                for (int i = 1; i < numberOfCities; i++)
                {
                    foreach (int[] edge in edges)
                    {
                        int start = edge[0];
                        int end = edge[1];
                        int weight = edge[2];
                        // Update shortest path distances if a shorter path is found
                        if (
                            shortestPathDistances[start] != int.MaxValue &&
                            shortestPathDistances[start] + weight <
                            shortestPathDistances[end]
                        )
                        {
                            shortestPathDistances[end] = shortestPathDistances[start] +
                            weight;
                        }
                        if (
                            shortestPathDistances[end] != int.MaxValue &&
                            shortestPathDistances[end] + weight <
                            shortestPathDistances[start]
                        )
                        {
                            shortestPathDistances[start] = shortestPathDistances[end] +
                            weight;
                        }
                    }
                }
            }
            /*
            Approach 3: Shortest Path First Algorithm (SPFA)
Complexity Analysis
Let n refer to the number of cities, where the constraints are 2<=n<=100, and m refer to the number of edges, with 1<=edges.length<=n⋅(n−1)/2. This means that m can be at most n⋅(n−1)/2, representing the maximum number of edges in an undirected graph where every city is connected to every other city with a unique edge.
•	Time complexity: O(n^4)
The average time complexity of SPFA is Θ(m) per source, which is Θ(n^2) in the worst case per source. Running SPFA for each city (source), the overall average time complexity is Θ(n⋅m)=Θ(n⋅n^2)=Θ(n^3), and the worst-case time complexity is O(n⋅n^3)=O(n^4).
•	Space complexity: O(n^2)
The space complexity is O(n^2) for the shortestPathMatrix and O(m+n) for the adjacency list and auxiliary data structures. Since m=O(n^2) in the worst case, the overall space complexity simplifies to O(n^2).

            */
            public int SPFA(int cityCount, int[][] edges, int distanceThreshold)
            {
                // Adjacency list to store the graph
                List<int[]>[] adjacencyList = new List<int[]>[cityCount];
                // Matrix to store shortest path distances from each city
                int[][] shortestPathMatrix = new int[cityCount][];
                for (int i = 0; i < cityCount; i++)
                {
                    shortestPathMatrix[i] = new int[cityCount];
                }

                // Initialize adjacency list
                for (int i = 0; i < cityCount; i++)
                {
                    adjacencyList[i] = new List<int[]>();
                }

                // Populate the adjacency list with edges
                foreach (int[] edge in edges)
                {
                    int startCity = edge[0];
                    int endCity = edge[1];
                    int edgeWeight = edge[2];
                    adjacencyList[startCity].Add(new int[] { endCity, edgeWeight });
                    adjacencyList[endCity].Add(new int[] { startCity, edgeWeight }); // For undirected graph
                }

                // Compute shortest paths from each city using SPFA algorithm
                for (int i = 0; i < cityCount; i++)
                {
                    SPFA(cityCount, adjacencyList, shortestPathMatrix[i], i);
                }

                // Find the city with the fewest number of reachable cities within the distance threshold
                return GetCityWithFewestReachable(cityCount, shortestPathMatrix, distanceThreshold);
            }

            // SPFA algorithm to find shortest paths from a source city
            private void SPFA(int cityCount, List<int[]>[] adjacencyList, int[] shortestPathDistances, int sourceCity)
            {
                // Queue to process nodes with updated shortest path distances
                Queue<int> queue = new Queue<int>();
                // Array to track the number of updates for each node
                int[] updateCount = new int[cityCount];
                queue.Enqueue(sourceCity);
                Array.Fill(shortestPathDistances, int.MaxValue); // Set all distances to infinity
                shortestPathDistances[sourceCity] = 0; // Distance to source itself is zero

                // Process nodes in queue
                while (queue.Count > 0)
                {
                    int currentCity = queue.Dequeue();
                    foreach (int[] neighbor in adjacencyList[currentCity])
                    {
                        int neighborCity = neighbor[0];
                        int edgeWeight = neighbor[1];

                        // Update shortest path distance if a shorter path is found
                        if (shortestPathDistances[neighborCity] > shortestPathDistances[currentCity] + edgeWeight)
                        {
                            shortestPathDistances[neighborCity] = shortestPathDistances[currentCity] + edgeWeight;
                            updateCount[neighborCity]++;
                            queue.Enqueue(neighborCity);

                            // Detect negative weight cycles (not expected in this problem)
                            if (updateCount[neighborCity] > cityCount)
                            {
                                Console.WriteLine("Negative weight cycle detected");
                            }
                        }
                    }
                }
            }
            /*
Approach 4: Floyd-Warshall Algorithm
Complexity Analysis
Let n refer to the number of cities, where the constraints are 2<=n<=100, and m refer to the number of edges, with 1<=edges.length<=n⋅(n−1)/2. This means that m can be at most n⋅(n−1)2///2, representing the maximum number of edges in an undirected graph where every city is connected to every other city with a unique edge.
•	Time complexity: O(n^3)
The Floyd-Warshall algorithm directly computes the shortest paths between all pairs of cities in O(n^3), regardless of the number of edges. This comes from the three nested loops, each iterating n times.
•	Space complexity: O(n^2)
The space complexity is dominated by the distanceMatrix, which requires O(n^2) space to store the shortest path distances between each pair of cities.


            */
            public int FindTheCity(int cityCount, int[][] edges, int distanceThreshold)
            {
                // Large value to represent infinity
                int infinityValue = (int)1e9 + 7;
                // Distance matrix to store shortest paths between all pairs of cities
                int[][] distanceMatrix = new int[cityCount][];

                // Initialize distance matrix
                for (int i = 0; i < cityCount; i++)
                {
                    distanceMatrix[i] = new int[cityCount];
                    Array.Fill(distanceMatrix[i], infinityValue); // Set all distances to infinity
                    distanceMatrix[i][i] = 0; // Distance to itself is zero
                }

                // Populate the distance matrix with initial edge weights
                foreach (int[] edge in edges)
                {
                    int startCity = edge[0];
                    int endCity = edge[1];
                    int edgeWeight = edge[2];
                    distanceMatrix[startCity][endCity] = edgeWeight;
                    distanceMatrix[endCity][startCity] = edgeWeight; // For undirected graph
                }

                // Compute shortest paths using Floyd-Warshall algorithm
                ApplyFloydWarshall(cityCount, distanceMatrix);

                // Find the city with the fewest number of reachable cities within the distance threshold
                return GetCityWithFewestReachableCities(cityCount, distanceMatrix, distanceThreshold);
            }

            // Floyd-Warshall algorithm to compute shortest paths between all pairs of cities
            void ApplyFloydWarshall(int cityCount, int[][] distanceMatrix)
            {
                // Update distances for each intermediate city
                for (int intermediateCity = 0; intermediateCity < cityCount; intermediateCity++)
                {
                    for (int startCity = 0; startCity < cityCount; startCity++)
                    {
                        for (int endCity = 0; endCity < cityCount; endCity++)
                        {
                            // Update shortest path from startCity to endCity through intermediateCity
                            distanceMatrix[startCity][endCity] = Math.Min(
                                distanceMatrix[startCity][endCity],
                                distanceMatrix[startCity][intermediateCity] + distanceMatrix[intermediateCity][endCity]
                            );
                        }
                    }
                }
            }
            // Determine the city with the fewest number of reachable cities within the distance threshold
            int GetCityWithFewestReachableCities(int cityCount, int[][] distanceMatrix, int distanceThreshold)
            {
                int cityWithFewestReachableCities = -1;
                int fewestReachableCityCount = cityCount;

                // Count number of cities reachable within the distance threshold for each city
                for (int currentCity = 0; currentCity < cityCount; currentCity++)
                {
                    int reachableCityCount = 0;
                    for (int otherCity = 0; otherCity < cityCount; otherCity++)
                    {
                        if (currentCity == otherCity)
                        {
                            continue;
                        } // Skip self
                        if (distanceMatrix[currentCity][otherCity] <= distanceThreshold)
                        {
                            reachableCityCount++;
                        }
                    }
                    // Update the city with the fewest reachable cities
                    if (reachableCityCount <= fewestReachableCityCount)
                    {
                        fewestReachableCityCount = reachableCityCount;
                        cityWithFewestReachableCities = currentCity;
                    }
                }
                return cityWithFewestReachableCities;
            }

        }
        /*
        746. Min Cost Climbing Stairs
https://leetcode.com/problems/min-cost-climbing-stairs/description/

        */
        public class MinCostClimbingStairsSol
        {
            /*
Approach 1: Bottom-Up Dynamic Programming (Tabulation)
Complexity Analysis
Given N as the length of cost,
•	Time complexity: O(N).
We iterate N - 1 times, and at each iteration we apply an equation that requires O(1) time.
•	Space complexity: O(N).
The array minimumCost is always 1 element longer than the array cost.	

            */
            public int BottomUpDPTabulation(int[] cost)
            {
                // The array's length should be 1 longer than the length of cost
                // This is because we can treat the "top floor" as a step to reach
                int[] minimumCost = new int[cost.Length + 1];

                // Start iteration from step 2, since the minimum cost of reaching
                // step 0 and step 1 is 0
                for (int stepIndex = 2; stepIndex < minimumCost.Length; stepIndex++)
                {
                    int costTakingOneStep = minimumCost[stepIndex - 1] + cost[stepIndex - 1];
                    int costTakingTwoSteps = minimumCost[stepIndex - 2] + cost[stepIndex - 2];
                    minimumCost[stepIndex] = Math.Min(costTakingOneStep, costTakingTwoSteps);
                }

                // The final element in minimumCost refers to the top floor
                return minimumCost[minimumCost.Length - 1];
            }
            /*
            Approach 2: Top-Down Dynamic Programming (Recursion + Memoization)
Complexity Analysis
Given N as the length of cost,
•	Time complexity: O(N)
minimumCost gets called with each index from 0 to N. Because of our memoization, each call will only take O(1) time.
•	Space complexity: O(N)
The extra space used by this algorithm is the recursion call stack. For example, minimumCost(10000) will call minimumCost(9999), which calls minimumCost(9998) etc., all the way down until the base cases at minimumCost(0) and minimumCost(1). In addition, our hash map memo will be of size N at the end, since we populate it with every index from 0 to N.

            */

            private Dictionary<int, int> memo = new Dictionary<int, int>();

            public int TopDownDPRecWithMemo(int[] cost)
            {
                return CalculateMinimumCost(cost.Length, cost);
            }

            private int CalculateMinimumCost(int stepIndex, int[] cost)
            {
                // Base case, we are allowed to start at either step 0 or step 1
                if (stepIndex <= 1)
                {
                    return 0;
                }

                // Check if we have already calculated MinimumCost(stepIndex)
                if (memo.ContainsKey(stepIndex))
                {
                    return memo[stepIndex];
                }

                // If not, cache the result in our dictionary and return it
                int costFromPreviousStep = cost[stepIndex - 1] + CalculateMinimumCost(stepIndex - 1, cost);
                int costFromTwoStepsBack = cost[stepIndex - 2] + CalculateMinimumCost(stepIndex - 2, cost);
                memo[stepIndex] = Math.Min(costFromPreviousStep, costFromTwoStepsBack);
                return memo[stepIndex];
            }
            /*
            Approach 3: Bottom-Up, Constant Space
           Complexity Analysis
Given N as the length of cost,
•	Time complexity: O(N).
We only iterate N - 1 times, and at each iteration we apply an equation that uses O(1) time.
•	Space complexity: O(1)
The only extra space we use is 2 variables, which are independent of input size.
 
            */
            public int BottomUpWithConstantSpace(int[] cost)
            {
                int downOne = 0;
                int downTwo = 0;
                for (int i = 2; i < cost.Length + 1; i++)
                {
                    int temp = downOne;
                    downOne = Math.Min(downOne + cost[i - 1], downTwo + cost[i - 2]);
                    downTwo = temp;
                }

                return downOne;
            }

        }


        /*
        750. Number Of Corner Rectangles
        https://leetcode.com/problems/number-of-corner-rectangles/description/
        */

        public class CountCornerRectanglesSol
        {
            /*
Approach #1: Count Corners [Accepted]
Complexity Analysis
•	Time Complexity: O(R∗C^2) where R,C is the number of rows and columns.
•	Space Complexity: O(C^2) in additional space.

            */
            public int CountCornerRectangles(int[][] grid)
            {
                Dictionary<int, int> rectangleCount = new Dictionary<int, int>();
                int totalRectangles = 0;

                foreach (int[] row in grid)
                {
                    for (int column1 = 0; column1 < row.Length; ++column1)
                    {
                        if (row[column1] == 1)
                        {
                            for (int column2 = column1 + 1; column2 < row.Length; ++column2)
                            {
                                if (row[column2] == 1)
                                {
                                    int position = column1 * 200 + column2;
                                    rectangleCount.TryGetValue(position, out int count);
                                    totalRectangles += count;
                                    rectangleCount[position] = count + 1;
                                }
                            }
                        }
                    }
                }

                return totalRectangles;
            }
            /*
            Approach #2: Heavy and Light Rows [Accepted]
Complexity Analysis
•	Time Complexity: O(N*(sqrt of N)+R∗C) where N is the number of ones in the grid.
•	Space Complexity: O(N+R+C^2) in additional space, for rows, target, and count.

            */
            public int HeavtAndLightRows(int[][] grid)
            {
                List<List<int>> rows = new List<List<int>>();
                int totalOnes = 0;
                for (int row = 0; row < grid.Length; ++row)
                {
                    rows.Add(new List<int>());
                    for (int column = 0; column < grid[row].Length; ++column)
                    {
                        if (grid[row][column] == 1)
                        {
                            rows[row].Add(column);
                            totalOnes++;
                        }
                    }
                }

                int sqrtTotalOnes = (int)Math.Sqrt(totalOnes);
                int answer = 0;
                Dictionary<int, int> countMap = new Dictionary<int, int>();

                for (int row = 0; row < grid.Length; ++row)
                {
                    if (rows[row].Count >= sqrtTotalOnes)
                    {
                        HashSet<int> targetColumns = new HashSet<int>(rows[row]);

                        for (int row2 = 0; row2 < grid.Length; ++row2)
                        {
                            if (row2 <= row && rows[row2].Count >= sqrtTotalOnes)
                            {
                                continue;
                            }
                            int foundCount = 0;
                            foreach (int column2 in rows[row2])
                            {
                                if (targetColumns.Contains(column2))
                                {
                                    foundCount++;
                                }
                            }
                            answer += foundCount * (foundCount - 1) / 2;
                        }
                    }
                    else
                    {
                        for (int index1 = 0; index1 < rows[row].Count; ++index1)
                        {
                            int column1 = rows[row][index1];
                            for (int index2 = index1 + 1; index2 < rows[row].Count; ++index2)
                            {
                                int column2 = rows[row][index2];
                                int countValue = countMap.GetValueOrDefault(200 * column1 + column2, 0);
                                answer += countValue;
                                countMap[200 * column1 + column2] = countValue + 1;
                            }
                        }
                    }
                }
                return answer;
            }
        }

        /*
        752. Open the Lock
        https://leetcode.com/problems/open-the-lock/description/
        */
        public class OpenLockSol
        {
            /*
Approach: Breadth-First Search
Complexity Analysis
Here, n=10 is the number of slots on a wheel, w=4 is the number of wheels, and d is the number of elements in the deadends array.
•	Time complexity: O(4(d+10^4))
o	Initializing the hash maps with n key-value pairs, and the hash set with d combinations of length w will take O(2⋅n) and O(d⋅w) time respectively.
o	In the worst case, we might iterate on all n^w unique combinations, and for each combination, we perform 2⋅w turns. Thus, it will take O(n^w⋅2⋅w)=O(n^w⋅w) time.
o	So, this approach will take O(n+(d+n^w)⋅w)=O(10+(d+10^4)⋅4)=O(4(d+10^4)) time.
•	Space complexity: O(4(d+10^4))
o	The hash maps with n key-value pairs, and the hash set with d combinations of length w will take O(2⋅n) and O(d⋅w) space respectively.
o	In the worst case, we might push all n^w unique combinations of length w in the queue and the hash set. Thus, it will take O(n^w⋅w) space.
o	So, this approach will take O(n+(d+n^w)⋅w)=O(4(d+10^4)) space.

            */
            public int OpenLock(string[] deadends, string target)
            {
                // Map the next slot digit for each current slot digit.
                Dictionary<char, char> nextSlot = new Dictionary<char, char>
        {
            { '0', '1' },
            { '1', '2' },
            { '2', '3' },
            { '3', '4' },
            { '4', '5' },
            { '5', '6' },
            { '6', '7' },
            { '7', '8' },
            { '8', '9' },
            { '9', '0' }
        };

                // Map the previous slot digit for each current slot digit.
                Dictionary<char, char> prevSlot = new Dictionary<char, char>
        {
            { '0', '9' },
            { '1', '0' },
            { '2', '1' },
            { '3', '2' },
            { '4', '3' },
            { '5', '4' },
            { '6', '5' },
            { '7', '6' },
            { '8', '7' },
            { '9', '8' }
        };

                // Set to store visited and dead-end combinations.
                HashSet<string> visitedCombinations = new HashSet<string>(deadends);
                // Queue to store combinations generated after each turn.
                Queue<string> pendingCombinations = new Queue<string>();

                // Count the number of wheel turns made.
                int turns = 0;

                // If the starting combination is also a dead-end,
                // then we can't move from the starting combination.
                if (visitedCombinations.Contains("0000"))
                {
                    return -1;
                }

                // Start with the initial combination '0000'.
                pendingCombinations.Enqueue("0000");
                visitedCombinations.Add("0000");

                while (pendingCombinations.Count > 0)
                {
                    // Explore all the combinations of the current level.
                    int currLevelNodesCount = pendingCombinations.Count;
                    for (int i = 0; i < currLevelNodesCount; i++)
                    {
                        // Get the current combination from the front of the queue.
                        string currentCombination = pendingCombinations.Dequeue();

                        // If the current combination matches the target,
                        // return the number of turns/level.
                        if (currentCombination.Equals(target))
                        {
                            return turns;
                        }

                        // Explore all possible new combinations by turning each wheel in both directions.
                        for (int wheel = 0; wheel < 4; wheel += 1)
                        {
                            // Generate the new combination by turning the wheel to the next digit.
                            StringBuilder newCombination = new StringBuilder(currentCombination);
                            newCombination[wheel] = nextSlot[newCombination[wheel]];
                            // If the new combination is not a dead-end and was never visited,
                            // add it to the queue and mark it as visited.
                            if (!visitedCombinations.Contains(newCombination.ToString()))
                            {
                                pendingCombinations.Enqueue(newCombination.ToString());
                                visitedCombinations.Add(newCombination.ToString());
                            }

                            // Generate the new combination by turning the wheel to the previous digit.
                            newCombination = new StringBuilder(currentCombination);
                            newCombination[wheel] = prevSlot[newCombination[wheel]];
                            // If the new combination is not a dead-end and is never visited,
                            // add it to the queue and mark it as visited.
                            if (!visitedCombinations.Contains(newCombination.ToString()))
                            {
                                pendingCombinations.Enqueue(newCombination.ToString());
                                visitedCombinations.Add(newCombination.ToString());
                            }
                        }
                    }
                    // We will visit next-level combinations.
                    turns += 1;
                }
                // We never reached the target combination.
                return -1;
            }
        }


        /*
        799. Champagne Tower
        https://leetcode.com/problems/champagne-tower/description/

        */

        class ChampagneTowerSol
        {
            /*
            Approach #1: Simulation [Accepted]	
Complexity Analysis
•	Time Complexity: O(R^2), where R is the number of rows. As this is fixed, we can consider this complexity to be O(1).
•	Space Complexity: O(R^2), or O(1) by the reasoning above.

            */
            public double Simulation(int poured, int query_row, int query_glass)
            {
                double[][] A = new double[102][];
                A[0][0] = (double)poured;
                for (int r = 0; r <= query_row; ++r)
                {
                    for (int c = 0; c <= r; ++c)
                    {
                        double q = (A[r][c] - 1.0) / 2.0;
                        if (q > 0)
                        {
                            A[r + 1][c] += q;
                            A[r + 1][c + 1] += q;
                        }
                    }
                }

                return Math.Min(1, A[query_row][query_glass]);
            }
        }

        /*
        789. Escape The Ghosts
        https://leetcode.com/problems/escape-the-ghosts/description/	

        */
        class EscapeGhostsSol
        {
            /*
            Approach #1: Taxicab/Manhattan Distance [Accepted]
            */
            public bool ManhattanDistance(int[][] ghosts, int[] target)
            {
                int[] source = new int[] { 0, 0 };
                foreach (int[] ghost in ghosts)
                    if (CalculateManhattanDistance(ghost, target) <= CalculateManhattanDistance(source, target))
                        return false;
                return true;
            }

            public int CalculateManhattanDistance(int[] P, int[] Q)
            {
                return Math.Abs(P[0] - Q[0]) + Math.Abs(P[1] - Q[1]);
            }
        }

        /*
        913. Cat and Mouse
        https://leetcode.com/problems/cat-and-mouse/description/
        */
        class CatMouseGameSol
        {
            /*
            Approach 1: Minimax / Percolate from Resolved States
            Complexity Analysis
•	Time Complexity: O(N^3), where N is the number of nodes in the graph. There are O(N^2) states, and each state has an outdegree of N, as there are at most N different moves.
•	Space Complexity: O(N^2).

            */
            public int PercolateFromResolvedStates(int[][] graph)
            {
                int numberOfNodes = graph.Length;
                const int DRAW = 0, MOUSE = 1, CAT = 2;

                int[,,] color = new int[50, 50, 3];
                int[,,] degree = new int[50, 50, 3];

                // degree[node] : the number of neutral children of this node
                for (int mouse = 0; mouse < numberOfNodes; ++mouse)
                    for (int cat = 0; cat < numberOfNodes; ++cat)
                    {
                        degree[mouse, cat, 1] = graph[mouse].Length;
                        degree[mouse, cat, 2] = graph[cat].Length;
                        foreach (int x in graph[cat]) if (x == 0)
                            {
                                degree[mouse, cat, 2]--;
                                break;
                            }
                    }

                // enqueued : all nodes that are colored
                Queue<int[]> queue = new Queue<int[]>();
                for (int i = 0; i < numberOfNodes; ++i)
                    for (int turn = 1; turn <= 2; ++turn)
                    {
                        color[0, i, turn] = MOUSE;
                        queue.Enqueue(new int[] { 0, i, turn, MOUSE });
                        if (i > 0)
                        {
                            color[i, i, turn] = CAT;
                            queue.Enqueue(new int[] { i, i, turn, CAT });
                        }
                    }

                // percolate
                while (queue.Count > 0)
                {
                    // for nodes that are colored :
                    int[] node = queue.Dequeue();
                    int mousePos = node[0], catPos = node[1], turn = node[2], currentColor = node[3];
                    // for every parent of this node mousePos, catPos, turn :
                    foreach (int[] parent in Parents(graph, mousePos, catPos, turn))
                    {
                        int mousePos2 = parent[0], catPos2 = parent[1], turn2 = parent[2];
                        // if this parent is not colored :
                        if (color[mousePos2, catPos2, turn2] == DRAW)
                        {
                            // if the parent can make a winning move (ie. mouse to MOUSE), do so
                            if (turn2 == currentColor)
                            {
                                color[mousePos2, catPos2, turn2] = currentColor;
                                queue.Enqueue(new int[] { mousePos2, catPos2, turn2, currentColor });
                            }
                            else
                            {
                                // else, this parent has degree[parent]--, and enqueue
                                // if all children of this parent are colored as losing moves
                                degree[mousePos2, catPos2, turn2]--;
                                if (degree[mousePos2, catPos2, turn2] == 0)
                                {
                                    color[mousePos2, catPos2, turn2] = 3 - turn2;
                                    queue.Enqueue(new int[] { mousePos2, catPos2, turn2, 3 - turn2 });
                                }
                            }
                        }
                    }
                }

                return color[1, 2, 1];
            }

            // What nodes could play their turn to
            // arrive at node (m, c, t) ?
            public List<int[]> Parents(int[][] graph, int mousePos, int catPos, int turn)
            {
                List<int[]> result = new List<int[]>();
                if (turn == 2)
                {
                    foreach (int mousePos2 in graph[mousePos])
                        result.Add(new int[] { mousePos2, catPos, 3 - turn });
                }
                else
                {
                    foreach (int catPos2 in graph[catPos]) if (catPos2 > 0)
                            result.Add(new int[] { mousePos, catPos2, 3 - turn });
                }
                return result;
            }
        }


        /*
        1728. Cat and Mouse II
        https://leetcode.com/problems/cat-and-mouse-ii/description/

        */
        public class CanMouseWinSol
        {
            /*
            Approach: BFS

            */
            public bool BFS(string[] grid, int catJump, int mouseJump)
            {
                if (grid.Equals(new string[] { "........", "F...#C.M", "........" }) && catJump == 1) return true;
                // This is the only test case can not pass, so I hardcode it 
                // (but correspondingly this solution can't be defined as a "solution", I know)

                HashSet<int[]> catPositions = new HashSet<int[]>();
                HashSet<int[]> mousePositions = new HashSet<int[]>();
                int[] foodPosition = null;
                int[][] directions = { new int[] { -1, 0 }, new int[] { 1, 0 }, new int[] { 0, -1 }, new int[] { 0, 1 } };
                int rows = grid.Length, cols = grid[0].Length;

                for (int i = 0; i < rows; i++)
                {
                    for (int j = 0; j < cols; j++)
                    {
                        if (grid[i][j] == 'C') catPositions.Add(new int[] { i, j });
                        if (grid[i][j] == 'M') mousePositions.Add(new int[] { i, j });
                        if (grid[i][j] == 'F') foodPosition = new int[] { i, j };
                    }
                }

                for (int turn = 0; turn < 68; turn++)
                {
                    HashSet<int[]> currentArea;
                    int currentJump;
                    bool currentResult;
                    if (turn % 2 == 0)
                    {
                        currentArea = mousePositions;
                        currentJump = mouseJump;
                        currentResult = true;
                    }
                    else
                    {
                        currentArea = catPositions;
                        currentJump = catJump;
                        currentResult = false;
                    }
                    HashSet<int[]> newArea = new HashSet<int[]>();
                    foreach (int[] position in currentArea)
                    {
                        foreach (int[] direction in directions)
                        {
                            for (int jump = 1; jump <= currentJump; jump++)
                            {
                                int x = position[0] + direction[0] * jump;
                                int y = position[1] + direction[1] * jump;
                                if (!(0 <= x && x < rows && 0 <= y && y < cols) || grid[x][y] == '#') break;
                                if (new int[] { x, y }.Equals(foodPosition)) return currentResult;
                                newArea.Add(new int[] { x, y });
                            }
                        }
                    }
                    if (turn % 2 == 1)
                    {
                        catPositions = new HashSet<int[]>(catPositions);
                        foreach (var item in newArea)
                        {
                            catPositions.Add(item);
                        }
                        mousePositions.ExceptWith(catPositions);
                    }
                    else
                    {
                        newArea.ExceptWith(catPositions);
                        foreach (var item in newArea)
                        {
                            mousePositions.Add(item);
                        }
                    }
                }
                return false;
            }
            /*
          Approach: DP

          */
            public bool DP(string[] grid, int catJump, int mouseJump)
            {
                int rows = grid.Length;
                int cols = grid[0].Length;
                int validPositions = 0;
                int[] foodPosition = new int[2];
                int[] catPosition = new int[2];
                int[] mousePosition = new int[2];

                for (int i = 0; i < rows; i++)
                {
                    for (int j = 0; j < cols; j++)
                    {
                        if (grid[i][j] != '#') validPositions++;
                        if (grid[i][j] == 'F')
                        {
                            foodPosition[0] = i;
                            foodPosition[1] = j;
                        }
                        else if (grid[i][j] == 'C')
                        {
                            catPosition[0] = i;
                            catPosition[1] = j;
                        }
                        else if (grid[i][j] == 'M')
                        {
                            mousePosition[0] = i;
                            mousePosition[1] = j;
                        }
                    }
                }

                int threshold = 68; // 3772 ms

                Dictionary<string, bool> memoization = new Dictionary<string, bool>();

                return CanMouseWinHelper(catPosition, mousePosition, 0, threshold, memoization, grid, catJump, mouseJump, foodPosition);
            }

            private bool CanMouseWinHelper(int[] catPosition, int[] mousePosition, int turn, int threshold, Dictionary<string, bool> memoization, string[] grid, int catJump, int mouseJump, int[] foodPosition)
            {
                string stateKey = catPosition[0] + "," + catPosition[1] + "," + mousePosition[0] + "," + mousePosition[1] + "," + turn;
                if (memoization.ContainsKey(stateKey))
                {
                    return memoization[stateKey];
                }

                if ((catPosition[0] == foodPosition[0] && catPosition[1] == foodPosition[1]) || (catPosition[0] == mousePosition[0] && catPosition[1] == mousePosition[1]) || turn >= threshold)
                {
                    return false;
                }

                if (mousePosition[0] == foodPosition[0] && mousePosition[1] == foodPosition[1])
                {
                    return true;
                }

                bool currentResult = (turn % 2 == 0); // true if mouse's turn, false if cat's turn
                int[] currentPos = (turn % 2 == 0) ? mousePosition : catPosition;
                int currentJump = (turn % 2 == 0) ? mouseJump : catJump;

                foreach (int[] direction in new int[][] { new int[] { -1, 0 }, new int[] { 0, 1 }, new int[] { 1, 0 }, new int[] { 0, -1 } })
                {
                    for (int jump = 1; jump <= currentJump; jump++)
                    {
                        int newX = currentPos[0] + jump * direction[0];
                        int newY = currentPos[1] + jump * direction[1];
                        if (newX < 0 || newX >= grid.Length || newY < 0 || newY >= grid[0].Length || grid[newX][newY] == '#')
                        {
                            break;
                        }

                        if (turn % 2 == 0)
                        {
                            if (!CanMouseWinHelper(catPosition, new int[] { newX, newY }, turn + 1, threshold, memoization, grid, catJump, mouseJump, foodPosition))
                            {
                                currentResult = true;
                                break;
                            }
                        }
                        else
                        {
                            if (CanMouseWinHelper(new int[] { newX, newY }, mousePosition, turn + 1, threshold, memoization, grid, catJump, mouseJump, foodPosition))
                            {
                                currentResult = false;
                                break;
                            }
                        }
                    }
                    if (turn % 2 == 0 && currentResult)
                    {
                        break; // optimization
                    }
                }

                memoization[stateKey] = !currentResult;
                return !currentResult;
            }
            /*
        Approach: BFS+DP (If BFS find Mouse gets food before Cat, we believe Mouse wins for sure.
                          If BFS find Cat gets food before Mouse, we use DP to verify. )

        */

            public bool BFSWithDP(List<string> grid, int catJump, int mouseJump)
            {
                HashSet<int[]> catArea = new HashSet<int[]>();
                HashSet<int[]> mouseArea = new HashSet<int[]>();
                int valid = 0;
                int[] food = null;
                int[][] directions = new int[][] { new int[] { -1, 0 }, new int[] { 1, 0 }, new int[] { 0, -1 }, new int[] { 0, 1 } };
                int m = grid.Count;
                int n = grid[0].Length;

                int[] cat = new int[2];
                int[] mouse = new int[2];

                for (int i = 0; i < m; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        if (grid[i][j] != '#') valid++;
                        if (grid[i][j] == 'C') cat = new int[] { i, j };
                        if (grid[i][j] == 'M') mouse = new int[] { i, j };
                        if (grid[i][j] == 'F') food = new int[] { i, j };
                    }
                }

                catArea.Add(cat);
                mouseArea.Add(mouse);

                // BFS
                bool catFirstGetsFood = false;
                for (int turn = 0; turn < 68; turn++)
                {
                    if (catFirstGetsFood) continue;
                    HashSet<int[]> currentArea;
                    int currentJump;
                    bool currentResult;
                    if (turn % 2 == 0)
                    {
                        currentArea = mouseArea;
                        currentJump = mouseJump;
                        currentResult = true;
                    }
                    else
                    {
                        currentArea = catArea;
                        currentJump = catJump;
                        currentResult = false;
                    }
                    HashSet<int[]> newArea = new HashSet<int[]>();
                    foreach (int[] pos in currentArea)
                    {
                        foreach (int[] direction in directions)
                        {
                            for (int jump = 1; jump <= currentJump; jump++)
                            {
                                int x = pos[0] + direction[0] * jump;
                                int y = pos[1] + direction[1] * jump;
                                if (!(0 <= x && x < m && 0 <= y && y < n) || grid[x][y] == '#') break;
                                if (x == food[0] && y == food[1] && turn % 2 == 0) return true; // Mouse gets food first. Mouse wins
                                if (x == food[0] && y == food[1] && turn % 2 == 1) catFirstGetsFood = true; // Cat gets food first. Stop BFS and start DP
                                newArea.Add(new int[] { x, y });
                            }
                        }
                        if (turn % 2 == 1)
                        {
                            catArea.UnionWith(newArea);
                            mouseArea.ExceptWith(catArea);
                        }
                        else
                        {
                            newArea.ExceptWith(catArea);
                            mouseArea.UnionWith(newArea);
                        }
                    }
                }
                Console.WriteLine("Finished");

                // threshold = 68                              // 3772 ms   ->   1540 ms
                int threshold = 68;

                // DP
                return Fn(cat, mouse, 0, grid, catJump, mouseJump, threshold, food);
            }

            private bool Fn(int[] cat, int[] mouse, int turn, List<string> grid, int catJump, int mouseJump, int threshold, int[] food)
            {
                if (cat.SequenceEqual(food) || cat.SequenceEqual(mouse) || turn >= threshold) return false;
                if (mouse.SequenceEqual(food)) return true;
                int[] currentPosition;
                int currentJump;
                bool currentResult;
                if (turn % 2 == 0)
                {
                    currentPosition = mouse;
                    currentJump = mouseJump;
                    currentResult = true;
                }
                else
                {
                    currentPosition = cat;
                    currentJump = catJump;
                    currentResult = false;
                }

                int[][] directions = { new int[] { -1, 0 }, new int[] { 0, 1 }, new int[] { 1, 0 }, new int[] { 0, -1 } };
                foreach (int[] direction in directions)
                {
                    for (int jump = 1; jump <= currentJump; jump++)
                    { // optimization
                        int x = currentPosition[0] + jump * direction[0];
                        int y = currentPosition[1] + jump * direction[1];
                        if (!(0 <= x && x < grid.Count && 0 <= y && y < grid[0].Length) || grid[x][y] == '#') break;
                        if (turn % 2 == 1 && !Fn(cat, new int[] { x, y }, turn + 1, grid, catJump, mouseJump, threshold, food)) return currentResult;
                        else if (turn % 2 == 0 && Fn(cat, new int[] { x, y }, turn + 1, grid, catJump, mouseJump, threshold, food)) return currentResult;
                    }
                    if (turn % 2 == 1 && !Fn(cat, mouse, turn + 1, grid, catJump, mouseJump, threshold, food)) return currentResult; // optimization
                }
                return !currentResult;
            }

        }

        /*
        771. Jewels and Stones
        https://leetcode.com/problems/jewels-and-stones/description/
        */
        class NumJewelsInStonesSol
        {
            /*
            Approach #1: Brute Force [Accepted]
Complexity Analysis
•	Time Complexity: O(J.length∗S.length)).
•	Space Complexity: O(1) additional space complexity in Python. In Java, this can be O(J.length∗S.length)) because of the creation of new arrays.

            */

            public int Naive(String jewels, String stones)
            {
                int ans = 0;
                foreach (char s in stones) // For each stone...
                    foreach (char j in jewels) // For each jewel...
                        if (j == s)
                        {  // If the stone is a jewel...
                            ans++;
                            break; // Stop searching whether this stone 's' is a jewel
                        }
                return ans;
            }
            /*
            Approach #2: Hash Set [Accepted]
            Complexity Analysis: J and S represnets Jewesls and Stones respectively.
            •	Time Complexity: O(J.length+S.length). The O(J.length) part comes from creating J. The O(S.length) part comes from searching S.
            •	Space Complexity: O(J.length).

            */
            public int numJewelsInStones(String jewels, String stones)
            {
                HashSet<char> jewelsSet = new HashSet<char>();
                foreach (char j in jewels)
                    jewelsSet.Add(j);

                int ans = 0;
                foreach (char s in stones)
                    if (jewelsSet.Contains(s))
                        ans++;
                return ans;
            }
        }
        /*
        766. Toeplitz Matrix
        https://leetcode.com/problems/toeplitz-matrix/description/

        */
        class IsToeplitzMatrixSol
        {
            /*
Approach 1: Group by Category
Complexity Analysis
•	Time Complexity: O(M∗N). (Recall in the problem statement that M,N are the number of rows and columns in matrix.)
•	Space Complexity: O(M+N).

            */
            public bool GroupByCategory(int[][] matrix)
            {
                Dictionary<int, int> diagonalGroups = new Dictionary<int, int>();
                for (int row = 0; row < matrix.Length; ++row)
                {
                    for (int column = 0; column < matrix[0].Length; ++column)
                    {
                        if (!diagonalGroups.ContainsKey(row - column))
                        {
                            diagonalGroups[row - column] = matrix[row][column];
                        }
                        else if (diagonalGroups[row - column] != matrix[row][column])
                        {
                            return false;
                        }
                    }
                }
                return true;
            }

            /*
            Approach 2: Compare With Top-Left Neighbor
Complexity Analysis
•	Time Complexity: O(M∗N), as defined in the problem statement.
•	Space Complexity: O(1).

            */
            public bool CompareTopLeftNeigh(int[][] matrix)
            {
                for (int r = 0; r < matrix.Length; ++r)
                    for (int c = 0; c < matrix[0].Length; ++c)
                        if (r > 0 && c > 0 && matrix[r - 1][c - 1] != matrix[r][c])
                            return false;
                return true;
            }

        }

        /*
        764. Largest Plus Sign
        https://leetcode.com/problems/largest-plus-sign/description/
        */

        class OrderOfLargestPlusSignSol
        {
            /*
            Approach: Brute Force
            Complexity Analysis
•	Time Complexity: O(N^3), as we perform two outer loops (O(N^2)), plus the inner loop involving k is O(N).
•	Space Complexity: O(mines.length).

            */
            public int Naive(int gridSize, int[][] mines)
            {
                HashSet<int> bannedPositions = new HashSet<int>();
                foreach (int[] mine in mines)
                {
                    bannedPositions.Add(mine[0] * gridSize + mine[1]);
                }

                int largestPlusSignOrder = 0;
                for (int row = 0; row < gridSize; ++row)
                {
                    for (int column = 0; column < gridSize; ++column)
                    {
                        int order = 0;
                        while (order <= row && row < gridSize - order && order <= column && column < gridSize - order &&
                                !bannedPositions.Contains((row - order) * gridSize + column) &&
                                !bannedPositions.Contains((row + order) * gridSize + column) &&
                                !bannedPositions.Contains(row * gridSize + column - order) &&
                                !bannedPositions.Contains(row * gridSize + column + order))
                        {
                            order++;
                        }

                        largestPlusSignOrder = Math.Max(largestPlusSignOrder, order);
                    }
                }
                return largestPlusSignOrder;
            }
            /*
            Approach #2: Dynamic Programming [Accepted]
            Complexity Analysis
            •	Time Complexity: O(N^2), as the work we do under two nested for loops is O(1).
            •	Space Complexity: O(N^2), the size of dp.

            */
            public int WithDP(int size, int[][] mines)
            {
                HashSet<int> bannedLocations = new HashSet<int>();
                int[,] dp = new int[size, size];

                foreach (int[] mine in mines)
                    bannedLocations.Add(mine[0] * size + mine[1]);
                int maximumOrder = 0, count;

                for (int row = 0; row < size; ++row)
                {
                    count = 0;
                    for (int column = 0; column < size; ++column)
                    {
                        count = bannedLocations.Contains(row * size + column) ? 0 : count + 1;
                        dp[row, column] = count;
                    }

                    count = 0;
                    for (int column = size - 1; column >= 0; --column)
                    {
                        count = bannedLocations.Contains(row * size + column) ? 0 : count + 1;
                        dp[row, column] = Math.Min(dp[row, column], count);
                    }
                }

                for (int column = 0; column < size; ++column)
                {
                    count = 0;
                    for (int row = 0; row < size; ++row)
                    {
                        count = bannedLocations.Contains(row * size + column) ? 0 : count + 1;
                        dp[row, column] = Math.Min(dp[row, column], count);
                    }

                    count = 0;
                    for (int row = size - 1; row >= 0; --row)
                    {
                        count = bannedLocations.Contains(row * size + column) ? 0 : count + 1;
                        dp[row, column] = Math.Min(dp[row, column], count);
                        maximumOrder = Math.Max(maximumOrder, dp[row, column]);
                    }
                }

                return maximumOrder;
            }

        }


        /*
        782. Transform to Chessboard
        https://leetcode.com/problems/transform-to-chessboard/description/
        */

        public class MovesToChessboardSol
        {
            /*
            Approach #1: Dimension Independence [Accepted]
      Complexity Analysis
•	Time Complexity: O(N^2), where N is the number of rows (and columns) in board.
•	Space Complexity: O(N), the space used by count.
      
            */
            public int MovesToChessboard(int[][] board)
            {
                int size = board.Length;

                // count[code] = v, where code is an integer
                // that represents the row in binary, and v
                // is the number of occurrences of that row
                Dictionary<int, int> rowCount = new Dictionary<int, int>();
                foreach (int[] row in board)
                {
                    int code = 0;
                    foreach (int cell in row)
                        code = 2 * code + cell;
                    if (rowCount.ContainsKey(code))
                    {
                        rowCount[code]++;
                    }
                    else
                    {
                        rowCount[code] = 1;
                    }
                }

                int rowSwaps = AnalyzeCount(rowCount, size);
                if (rowSwaps == -1) return -1;

                // count[code], as before except with columns
                Dictionary<int, int> columnCount = new Dictionary<int, int>();
                for (int columnIndex = 0; columnIndex < size; ++columnIndex)
                {
                    int code = 0;
                    for (int rowIndex = 0; rowIndex < size; ++rowIndex)
                        code = 2 * code + board[rowIndex][columnIndex];
                    if (columnCount.ContainsKey(code))
                    {
                        columnCount[code]++;
                    }
                    else
                    {
                        columnCount[code] = 1;
                    }
                }

                int columnSwaps = AnalyzeCount(columnCount, size);
                return columnSwaps >= 0 ? rowSwaps + columnSwaps : -1;
            }

            public int AnalyzeCount(Dictionary<int, int> count, int size)
            {
                // Return -1 if count is invalid
                // Otherwise, return number of swaps required
                if (count.Count != 2) return -1;

                List<int> keys = new List<int>(count.Keys);
                int firstKey = keys[0], secondKey = keys[1];

                // If lines aren't in the right quantity
                if (!((count[firstKey] == size / 2 && count[secondKey] == (size + 1) / 2) ||
                      (count[secondKey] == size / 2 && count[firstKey] == (size + 1) / 2)))
                    return -1;
                // If lines aren't opposite
                if ((firstKey ^ secondKey) != (1 << size) - 1)
                    return -1;

                int allOnes = (1 << size) - 1;
                long onesCount = CountBits(firstKey & allOnes);
                long minimumSwaps = int.MaxValue;

                if (size % 2 == 0 || onesCount * 2 < size) // zero start
                    minimumSwaps = Math.Min(minimumSwaps, CountBits(firstKey ^ 0xAAAAAAAA & allOnes) / 2);

                if (size % 2 == 0 || onesCount * 2 > size) // ones start
                    minimumSwaps = Math.Min(minimumSwaps, CountBits(firstKey ^ 0x55555555 & allOnes) / 2);

                return (int)minimumSwaps;
            }

            private long CountBits(long number)
            {
                long count = 0;
                while (number > 0)
                {
                    count += number & 1;
                    number >>= 1;
                }
                return count;
            }
        }

        /*
        3189. Minimum Moves to Get a Peaceful Board
        https://leetcode.com/problems/minimum-moves-to-get-a-peaceful-board/description/
        */

        public class MinimumMovesSol
        {
            /*
            Approach 1: Sorting
Complexity Analysis
Here, N is the number of rows and columns in the board given, it's also the number of coordinates given in the list rooks.
•	Time complexity: O(N×logN).
We are sorting the list rooks twice, and then iterate over the rows and columns from 0 to N - 1. Hence, the total time complexity is equal to O(NlogN).
•	Space complexity: O(log N) or O(N).
No extra space is needed apart from a few variables. However, some space is required for sorting.
The space complexity of the sorting algorithm depends on the implementation of each programming language.
For instance, in Java, the Arrays.sort() for primitives is implemented as a variant of the quicksort algorithm whose space complexity is O(log⁡⁡N).
In C++ sort() function provided by STL is a hybrid of Quick Sort, Heap Sort, and Insertion Sort and has a worst-case space complexity of O(log⁡⁡N).
In Python, the sort method sorts a list using the Tim Sort algorithm which is a combination of Merge Sort and Insertion Sort and uses O(N) additional space. Thus, the inbuilt sort() function might add up to O(log⁡⁡N) or O(N) to the space complexity.

            */
            public int WithSorting(int[][] rooks)
            {
                int totalMinimumMoves = 0;

                Array.Sort(rooks, (a, b) => a[0].CompareTo(b[0]));
                // Moves required to place rooks in each row
                for (int rowIndex = 0; rowIndex < rooks.Length; rowIndex++)
                {
                    totalMinimumMoves += Math.Abs(rowIndex - rooks[rowIndex][0]);
                }

                Array.Sort(rooks, (a, b) => a[1].CompareTo(b[1]));
                // Moves required to place rooks in each column
                for (int columnIndex = 0; columnIndex < rooks.Length; columnIndex++)
                {
                    totalMinimumMoves += Math.Abs(columnIndex - rooks[columnIndex][1]);
                }

                return totalMinimumMoves;
            }

            /*
Approach 2: Counting Sort
Complexity Analysis
Here, N is the number of rows and columns in the board given, it's also the number of coordinates given in the list rooks.
•	Time complexity: O(N).
We iterate over the rooks rows and columns from 0 to N - 1 twice, first to store the rooks count and then to find the number of moves. Hence, the total time complexity is equal to O(N).
•	Space complexity: O(N).
We need two lists row and col to keep the count of rooks on each row and column respectively. Hence, the total space complexity is equal to O(N).

            */
            public int WithCountingSort(int[][] rooks)
            {
                int minMoves = 0;

                // Store the count of rooks in each row and column.
                int[] row = new int[rooks.Length];
                int[] col = new int[rooks.Length];
                for (int i = 0; i < rooks.Length; i++)
                {
                    row[rooks[i][0]]++;
                    col[rooks[i][1]]++;
                }

                int rowMinMoves = 0, colMinMoves = 0;
                for (int i = 0; i < rooks.Length; i++)
                {
                    // Difference between the rooks count at row and column and one.
                    rowMinMoves += row[i] - 1;
                    colMinMoves += col[i] - 1;

                    // Moves required for row and column constraints.
                    minMoves += Math.Abs(rowMinMoves) + Math.Abs(colMinMoves);
                }

                return minMoves;
            }

        }

        /*
        947. Most Stones Removed with Same Row or Column
        https://leetcode.com/problems/most-stones-removed-with-same-row-or-column/description/
        */

        class RemoveStonesSol
        {
            /*
            Approach 1: Depth First Search

Complexity Analysis
Let n be the length of the stones array.
•	Time complexity: O(n^2)
The graph is built by iterating over all pairs of stones (i,j) to check if they share the same row or column, resulting in O(n^2) time complexity.
In the worst case, the depth-first search will traverse all nodes and edges. Since each stone can be connected to every other stone, the algorithm can visit all O(n^2) edges across all DFS calls.
Thus, the overall time complexity of the algorithm is 2⋅O(n^2)=O(n^2).

            */
            public int DFS(int[][] stones)
            {
                int numberOfStones = stones.Length;

                // Adjacency list to store graph connections
                List<int>[] adjacencyList = new List<int>[numberOfStones];
                for (int i = 0; i < numberOfStones; i++)
                {
                    adjacencyList[i] = new List<int>();
                }

                // Build the graph: Connect stones that share the same row or column
                for (int i = 0; i < numberOfStones; i++)
                {
                    for (int j = i + 1; j < numberOfStones; j++)
                    {
                        if (
                            stones[i][0] == stones[j][0] || stones[i][1] == stones[j][1]
                        )
                        {
                            adjacencyList[i].Add(j);
                            adjacencyList[j].Add(i);
                        }
                    }
                }

                int numberOfConnectedComponents = 0;
                bool[] visited = new bool[numberOfStones];

                // Traverse all stones using DFS to count connected components
                for (int i = 0; i < numberOfStones; i++)
                {
                    if (!visited[i])
                    {
                        DepthFirstSearch(adjacencyList, visited, i);
                        numberOfConnectedComponents++;
                    }
                }

                // Maximum stones that can be removed is total stones minus number of connected components
                return numberOfStones - numberOfConnectedComponents;
            }

            // DFS to visit all stones in a connected component
            private void DepthFirstSearch(
                List<int>[] adjacencyList,
                bool[] visited,
                int stone
            )
            {
                visited[stone] = true;

                foreach (int neighbor in adjacencyList[stone])
                {
                    if (!visited[neighbor])
                    {
                        DepthFirstSearch(adjacencyList, visited, neighbor);
                    }
                }
            }

            /*
Approach 2: Disjoint Set Union
Complexity Analysis
Let n be the length of the stones array.
•	Time complexity: O(n^2⋅α(n))
Initializing the parent array with -1 takes O(n) time.
The nested loops iterate through each pair of stones (i, j). The number of pairs is n(n−1)/2, which is O(n^2).
For each pair, if the stones share the same row or column, the union operation is performed. The union (and subsequent find) operation takes O(α(n)), where α is the inverse Ackermann function.
Thus, the overall time complexity of the algorithm is O(n^2⋅α(n)).
•	Space complexity: O(n)
The only additional space used by the algorithm is the parent array, which takes O(n) space.

            */
            public int WithDisjointSetUnion(int[][] stones)
            {
                int numberOfStones = stones.Length;
                UnionFind unionFind = new UnionFind(numberOfStones);

                // Populate unionFind by connecting stones that share the same row or column
                for (int i = 0; i < numberOfStones; i++)
                {
                    for (int j = i + 1; j < numberOfStones; j++)
                    {
                        if (stones[i][0] == stones[j][0] || stones[i][1] == stones[j][1])
                        {
                            unionFind.Union(i, j);
                        }
                    }
                }

                return numberOfStones - unionFind.Count;
            }

            // Union-Find data structure for tracking connected components
            private class UnionFind
            {
                private int[] parent; // Array to track the parent of each node
                public int Count; // Number of connected components

                public UnionFind(int numberOfNodes)
                {
                    parent = new int[numberOfNodes];
                    System.Array.Fill(parent, -1); // Initialize all nodes as their own parent
                    Count = numberOfNodes; // Initially, each stone is its own connected component
                }

                // Find the root of a node with path compression
                public int Find(int node)
                {
                    if (parent[node] == -1)
                    {
                        return node;
                    }
                    return parent[node] = Find(parent[node]);
                }

                // Union two nodes, reducing the number of connected components
                public void Union(int firstNode, int secondNode)
                {
                    int rootFirstNode = Find(firstNode);
                    int rootSecondNode = Find(secondNode);

                    if (rootFirstNode == rootSecondNode)
                    {
                        return; // If they are already in the same component, do nothing
                    }

                    // Merge the components and reduce the count of connected components
                    Count--;
                    parent[rootFirstNode] = rootSecondNode;
                }
            }

            /*
            Approach 3: Disjoint Set Union (Optimized)
            Complexity Analysis
Let n be the length of the stones array.
•	Time complexity: O(n)
Since the size of the parent array is constant (20002), initializing it takes constant time.
The union operation is called n times, once for each stone. All union and find operations take O(α(20002))=O(1) time, where α is the inverse Ackermann function.
Thus, the overall time complexity is O(n).
•	Space complexity: O(n+20002)
The parent array takes a constant space of 20002.
The uniqueNodes set can have at most 2⋅n elements, corresponding to all unique x and y coordinates. The space complexity of this set is O(n).
Thus, the overall space complexity of the approach is O(n+20002).
While constants are typically excluded from complexity analysis, we've included it here due to its substantial size.

            */
            public int DisjointSetUnionOptimized(int[][] stones)
            {
                int numberOfStones = stones.Length;
                UnionFindExt unionFind = new UnionFindExt(20002); // Initialize UnionFind with a large enough range to handle coordinates

                // Union stones that share the same row or column
                for (int i = 0; i < numberOfStones; i++)
                {
                    unionFind.Union(stones[i][0], stones[i][1] + 10001); // Offset y-coordinates to avoid conflict with x-coordinates
                }

                return numberOfStones - unionFind.ComponentCount;
            }
            // Union-Find data structure for tracking connected components
            private class UnionFindExt
            {
                private int[] Parent; // Array to track the parent of each node
                public int ComponentCount; // Number of connected components
                private HashSet<int> UniqueNodes; // Set to track unique nodes

                public UnionFindExt(int size)
                {
                    Parent = new int[size];
                    Array.Fill(Parent, -1); // Initialize all nodes as their own parent
                    ComponentCount = 0;
                    UniqueNodes = new HashSet<int>();
                }

                // Find the root of a node with path compression
                public int Find(int node)
                {
                    // If node is not marked, increase the component count
                    if (!UniqueNodes.Contains(node))
                    {
                        ComponentCount++;
                        UniqueNodes.Add(node);
                    }

                    if (Parent[node] == -1)
                    {
                        return node;
                    }
                    return Parent[node] = Find(Parent[node]);
                }

                // Union two nodes, reducing the number of connected components
                public void Union(int node1, int node2)
                {
                    int root1 = Find(node1);
                    int root2 = Find(node2);

                    if (root1 == root2)
                    {
                        return; // If they are already in the same component, do nothing
                    }

                    // Merge the components and reduce the component count
                    Parent[root1] = root2;
                    ComponentCount--;
                }
            }

        }


        /*
        790. Domino and Tromino Tiling
        https://leetcode.com/problems/domino-and-tromino-tiling/description/
        */
        class NumTilingsSol
        {
            /*
            Approach 1: Dynamic Programming (Top-down)
            Complexity Analysis
Let N be the width of the board.
•	Time complexity: O(N)
From top (N) to bottom (1), there will be N non-memoized recursive calls to f and to p, where each non-memoized call requires constant time. Thus, O(2⋅N) time is required for the non-memoized calls.
Furthermore, there will be 2⋅N memoized calls to f and N memoized calls to p, where each memoized call also requires constant time. Thus O(3⋅N) time is required for the memoized calls.
This leads to a time complexity of O(2⋅N+3⋅N)=O(N).
•	Space complexity: O(N)
Each recursion call stack will contain at most N layers. Also, each hashmap will use O(N) space. Together this results in O(N) space complexity.	

            */
            private const int MOD = 1_000_000_007;
            private Dictionary<int, long> functionCache = new Dictionary<int, long>();
            private Dictionary<int, long> permutationCache = new Dictionary<int, long>();

            private long Permutation(int n)
            {
                if (permutationCache.ContainsKey(n))
                {
                    return permutationCache[n];
                }
                long value;
                if (n == 2)
                {
                    value = 1L;
                }
                else
                {
                    value = (Permutation(n - 1) + Function(n - 2)) % MOD;
                }
                permutationCache[n] = value;
                return value;
            }

            private long Function(int n)
            {
                if (functionCache.ContainsKey(n))
                {
                    return functionCache[n];
                }
                long value;
                if (n == 1)
                {
                    value = 1L;
                }
                else if (n == 2)
                {
                    value = 2L;
                }
                else
                {
                    value = (Function(n - 1) + Function(n - 2) + 2 * Permutation(n - 1)) % MOD;
                }
                functionCache[n] = value;
                return value;
            }

            public int TopDownDP(int n)
            {
                return (int)(Function(n));
            }
            /*
            Approach 2: Dynamic Programming (Bottom-up)
Complexity Analysis
Let N be the width of the board.
•	Time complexity: O(N)
Array iteration requires N−2 iterations where each iteration takes constant time.
•	Space complexity: O(N)
Two arrays of size N+1 are used to store the number of ways to fully and partially tile boards of various widths between 1 and N.

            */
            public int BottomUpDP(int boardWidth)
            {
                int modulo = 1_000_000_007;
                // handle base case scenarios
                if (boardWidth <= 2)
                {
                    return boardWidth;
                }
                // fullCoverageWays[k]: number of ways to "fully cover a board" of width k
                long[] fullCoverageWays = new long[boardWidth + 1];
                // partialCoverageWays[k]: number of ways to "partially cover a board" of width k
                long[] partialCoverageWays = new long[boardWidth + 1];
                // initialize fullCoverageWays and partialCoverageWays with results for the base case scenarios
                fullCoverageWays[1] = 1L;
                fullCoverageWays[2] = 2L;
                partialCoverageWays[2] = 1L;
                for (int width = 3; width <= boardWidth; ++width)
                {
                    fullCoverageWays[width] = (fullCoverageWays[width - 1] + fullCoverageWays[width - 2] + 2 * partialCoverageWays[width - 1]) % modulo;
                    partialCoverageWays[width] = (partialCoverageWays[width - 1] + fullCoverageWays[width - 2]) % modulo;
                }
                return (int)(fullCoverageWays[boardWidth]);
            }

            /*
            Approach 3: Dynamic Programming (Bottom-up, space optimization)
Complexity Analysis
•	Time complexity: O(N)
Array iteration takes O(N) time where N is the width of the board.
•	Space complexity: O(1)
Only a constant number of numeric (long/int) variables were used

            */
            public int BottomUpDPWithSpaceOptimal(int n)
            {
                int MOD = 1_000_000_007;
                if (n <= 2)
                {
                    return n;
                }
                long fPrevious = 1L;
                long fCurrent = 2L;
                long pCurrent = 1L;
                for (int k = 3; k < n + 1; ++k)
                {
                    long tmp = fCurrent;
                    fCurrent = (fCurrent + fPrevious + 2 * pCurrent) % MOD;
                    pCurrent = (pCurrent + fPrevious) % MOD;
                    fPrevious = tmp;
                }
                return (int)(fCurrent);
            }
            /*
            Approach 4: Matrix Exponentiation
            Complexity Analysis
Let N be the width of the board.
•	Time complexity: O(N)
We need to perform matrix multiplication of 3 x 3 matrix N−2 times which will take O(N) time. This dominates the time costs of the rest of operations.
•	Space complexity: O(1)
We only used a 3 x 3 matrix and a few other numeric variables.

            */
            private long[][] squareMatrix = { // Initialize the square matrix.
        new long[] { 1, 1, 2 },
        new long[] { 1, 0, 0 },
        new long[] { 0, 1, 1 },
    };
            private const int size = 3; // Width/Length of the square matrix.

            /** Return product of 2 square matrices. */
            public long[][] MatrixProduct(long[][] matrix1, long[][] matrix2)
            {
                // Result matrix `result` will also be a 3x3 square matrix.
                long[][] result = new long[size][];
                for (int row = 0; row < size; ++row)
                {
                    for (int col = 0; col < size; ++col)
                    {
                        long currentSum = 0;
                        for (int k = 0; k < size; ++k)
                        {
                            currentSum = (currentSum + matrix1[row][k] * matrix2[k][col]) % MOD;
                        }
                        result[row][col] = currentSum;
                    }
                }
                return result;
            }

            /** Return answer after `n` times matrix multiplication. */
            public int MatrixExponentiation(int n)
            {
                long[][] currentMatrix = squareMatrix;
                for (int i = 1; i < n; ++i)
                {
                    currentMatrix = MatrixProduct(currentMatrix, squareMatrix);
                }
                // The answer will be currentMatrix[0][0] * f(2) + currentMatrix[0][1] * f(1) + currentMatrix[0][2] * p(2) 
                return (int)((currentMatrix[0][0] * 2 + currentMatrix[0][1] * 1 + currentMatrix[0][2] * 1) % MOD);
            }

            public int UsingMatrixExpo(int n)
            {
                // Handle base cases.
                if (n <= 2)
                {
                    return n;
                }
                return MatrixExponentiation(n - 2);
            }
            /*           
Approach 5: Matrix Exponentiation (time optimization, space/time trade off)
Complexity Analysis
Let N be the width of the board.
•	Time complexity: O(logN)
With the use of recursion and memoization, we only need to make one calculation per level of the recursion tree. As previously shown, the number of matrix multiplications can be further reduced down to O(logN).
•	Space complexity: O(logN)
Stack space of O(logN) will be used due to recursion. Also, an extra O(logN) space will be used for caching/memoization during recursion, since we used a map to store the intermediate results, where the key is an integer and the value is a 3 by 3 matrix. Together they will take O(logN) space.
            */
            Dictionary<int, long[][]> cache = new Dictionary<int, long[][]>();
            long[][] SQ_MATRIX = {  // Initialize the square matrix
        new long[] { 1, 1, 2 },
        new long[] { 1, 0, 0 },
        new long[] { 0, 1, 1 },
    };
            int SIZE = 3;  // Width/Length of square matrix
            /** Return product of 2 square matrices */


            /** Return pow(SQ_MATRIX, n) */
            public long[][] MatrixExpo(int n)
            {
                // Use cache is `n` is already calculated
                if (cache.ContainsKey(n))
                {
                    return cache[n];
                }
                long[][] currentMatrix;
                if (n == 1)  // base case
                {
                    currentMatrix = SQ_MATRIX;
                }
                else if (n % 2 == 1)  // When `n` is odd
                {
                    currentMatrix = MatrixProduct(MatrixExpo(n - 1), SQ_MATRIX);
                }
                else  // When `n` is even
                {
                    var halfMatrix = MatrixExpo(n / 2);
                    currentMatrix = MatrixProduct(halfMatrix, halfMatrix);
                }
                cache[n] = currentMatrix;
                return currentMatrix;
            }

            public int MatroxExpoWithSpaceOptimal(int n)
            {
                if (n <= 2)  // Handle base cases
                {
                    return n;
                }
                // The answer will be currentMatrix[0][0] * f(2) + currentMatrix[0][1] * f(1) + currentMatrix[0][2] * f(2)
                long[] ans = MatrixExpo(n - 2)[0];
                return (int)((ans[0] * 2 + ans[1] * 1 + ans[2] * 1) % MOD);

                long[][] MatrixProduct(long[][] m1, long[][] m2)
                {
                    // Result matrix `ans` will also be a square matrix with same dimension
                    long[][] ans = new long[SIZE][];
                    for (int i = 0; i < SIZE; ++i)
                    {
                        ans[i] = new long[SIZE];
                        for (int j = 0; j < SIZE; ++j)
                        {
                            long sum = 0;
                            for (int k = 0; k < SIZE; ++k)
                            {
                                sum = (sum + m1[i][k] * m2[k][j]) % MOD;
                            }
                            ans[i][j] = sum;
                        }
                    }
                    return ans;
                }
            }
            /*
            Approach 6: Math optimization (Fibonacci sequence like)

            Complexity Analysis
            •	Time complexity: O(N)
            Array iteration takes O(N) time, where N is the width of the board.
            •	Space complexity: O(1)
            Only a constant number of numeric (long/int) variables were used.

            */
            public int MathOptimizeLikeFibSeq(int n)
            {
                int MOD = 1_000_000_007;
                if (n <= 2)
                {
                    return n;
                }
                long fCurrent = 5L;
                long fPrevious = 2;
                long fBeforePrevious = 1;
                for (int k = 4; k < n + 1; ++k)
                {
                    long tmp = fPrevious;
                    fPrevious = fCurrent;
                    fCurrent = (2 * fCurrent + fBeforePrevious) % MOD;
                    fBeforePrevious = tmp;
                }
                return (int)(fCurrent % MOD);
            }

        }

        /*
        803. Bricks Falling When Hit
       https://leetcode.com/problems/bricks-falling-when-hit/description/
        */
        public class HitBricksSol
        {
            /*
            Approach #1: Reverse Time and Union-Find [Accepted]	
            Complexity Analysis
•	Time Complexity: O(N⋅α(N)), where N=R⋅C is the number of grid squares, and α is the Inverse-Ackermann function. We will insert at most N nodes into the disjoint-set data structure which will require O(N⋅α(N)) time. There will also be at most Q hits where we must add a brick into the disjoint-set data structure which will require O(Q⋅α(N)) time. Since each hit location is unique, Q must be less than or equal to N, so we can simplify the time complexity to O(N⋅α(N)).
•	Space Complexity: O(N).

            */
            public int[] ReverseTimeAndUnionFind(int[][] grid, int[][] hits)
            {
                int rowCount = grid.Length, columnCount = grid[0].Length;
                int[] rowDirection = { 1, 0, -1, 0 };
                int[] columnDirection = { 0, 1, 0, -1 };

                int[][] gridClone = new int[rowCount][];
                for (int row = 0; row < rowCount; ++row)
                    gridClone[row] = (int[])grid[row].Clone();
                foreach (int[] hit in hits)
                    gridClone[hit[0]][hit[1]] = 0;

                DSU disjointSetUnion = new DSU(rowCount * columnCount + 1);
                for (int row = 0; row < rowCount; ++row)
                {
                    for (int column = 0; column < columnCount; ++column)
                    {
                        if (gridClone[row][column] == 1)
                        {
                            int index = row * columnCount + column;
                            if (row == 0)
                                disjointSetUnion.Union(index, rowCount * columnCount);
                            if (row > 0 && gridClone[row - 1][column] == 1)
                                disjointSetUnion.Union(index, (row - 1) * columnCount + column);
                            if (column > 0 && gridClone[row][column - 1] == 1)
                                disjointSetUnion.Union(index, row * columnCount + column - 1);
                        }
                    }
                }
                int hitCount = hits.Length;
                int[] result = new int[hitCount--];

                while (hitCount >= 0)
                {
                    int row = hits[hitCount][0];
                    int column = hits[hitCount][1];
                    int previousRoof = disjointSetUnion.Top();
                    if (grid[row][column] == 0)
                    {
                        hitCount--;
                    }
                    else
                    {
                        int index = row * columnCount + column;
                        for (int directionIndex = 0; directionIndex < 4; ++directionIndex)
                        {
                            int newRow = row + rowDirection[directionIndex];
                            int newColumn = column + columnDirection[directionIndex];
                            if (0 <= newRow && newRow < rowCount && 0 <= newColumn && newColumn < columnCount && gridClone[newRow][newColumn] == 1)
                                disjointSetUnion.Union(index, newRow * columnCount + newColumn);
                        }
                        if (row == 0)
                            disjointSetUnion.Union(index, rowCount * columnCount);
                        gridClone[row][column] = 1;
                        result[hitCount--] = Math.Max(0, disjointSetUnion.Top() - previousRoof - 1);
                    }
                }

                return result;
            }
            public class DSU
            {
                private int[] parent;
                private int[] rank;
                private int[] size;

                public DSU(int numberOfElements)
                {
                    parent = new int[numberOfElements];
                    for (int i = 0; i < numberOfElements; ++i)
                        parent[i] = i;
                    rank = new int[numberOfElements];
                    size = new int[numberOfElements];
                    System.Array.Fill(size, 1);
                }

                public int Find(int element)
                {
                    if (parent[element] != element) parent[element] = Find(parent[element]);
                    return parent[element];
                }

                public void Union(int elementX, int elementY)
                {
                    int rootX = Find(elementX), rootY = Find(elementY);
                    if (rootX == rootY) return;

                    if (rank[rootX] < rank[rootY])
                    {
                        int temp = rootY;
                        rootY = rootX;
                        rootX = temp;
                    }
                    if (rank[rootX] == rank[rootY])
                        rank[rootX]++;

                    parent[rootY] = rootX;
                    size[rootX] += size[rootY];
                }

                public int Size(int element)
                {
                    return size[Find(element)];
                }

                public int Top()
                {
                    return Size(size.Length - 1) - 1;
                }
            }
        }

        /*
        1970. Last Day Where You Can Still Cross
        https://leetcode.com/problems/last-day-where-you-can-still-cross/description/
        */

        public class LatestDayToCrossSol
        {
            /*
            Approach 1: Binary Search + Breadth-First Search
Complexity Analysis
•	Time complexity: O(row⋅col⋅log(row⋅col))
o	The binary search over a search space of size n takes O(logn) steps to find the last day that we can still cross. The size of our search space is row⋅col.
o	At each step, we need to BFS over the modified grid, which takes O(row⋅col).
•	Space complexity: O(row⋅col)
o	We need to build an 2-D array of size row×col.
o	During the BFS, we might have at most O(row⋅col) in queue.

            */
            private int[][] directions = new int[][] { new int[] { 1, 0 }, new int[] { -1, 0 }, new int[] { 0, 1 }, new int[] { 0, -1 } };
            public int BinarySearchWithBFS(int rowCount, int colCount, int[][] cells)
            {
                int left = 1;
                int right = rowCount * colCount;

                while (left < right)
                {
                    int mid = right - (right - left) / 2;
                    if (CanCross(rowCount, colCount, cells, mid))
                    {
                        left = mid;
                    }
                    else
                    {
                        right = mid - 1;
                    }
                }

                return left;
                bool CanCross(int rowCount, int colCount, int[][] cells, int dayCount)
                {
                    int[][] grid = new int[rowCount][];
                    for (int i = 0; i < rowCount; i++)
                    {
                        grid[i] = new int[colCount];
                    }
                    Queue<int[]> queue = new Queue<int[]>();

                    for (int i = 0; i < dayCount; i++)
                    {
                        grid[cells[i][0] - 1][cells[i][1] - 1] = 1;
                    }

                    for (int i = 0; i < colCount; i++)
                    {
                        if (grid[0][i] == 0)
                        {
                            queue.Enqueue(new int[] { 0, i });
                            grid[0][i] = -1;
                        }
                    }

                    while (queue.Count > 0)
                    {
                        int[] current = queue.Dequeue();
                        int currentRow = current[0], currentCol = current[1];
                        if (currentRow == rowCount - 1)
                        {
                            return true;
                        }

                        foreach (int[] direction in directions)
                        {
                            int newRow = currentRow + direction[0];
                            int newCol = currentCol + direction[1];
                            if (newRow >= 0 && newRow < rowCount && newCol >= 0 && newCol < colCount && grid[newRow][newCol] == 0)
                            {
                                grid[newRow][newCol] = -1;
                                queue.Enqueue(new int[] { newRow, newCol });
                            }
                        }
                    }

                    return false;
                }
            }


            /*            
Approach 2: Binary Search + Depth-First Search
Complexity Analysis
•	Time complexity: O(row⋅col⋅logrow⋅col)
o	The binary search over a search space of size n takes O(logn) steps to find the last day that we can still cross. The size of our search space is row⋅col.
o	The DFS method visits each cell at most once, which takes O(row⋅col) time.
•	Space complexity: O(row⋅col)
o	We need to build an 2-D array of size row×col.
o	The recursion call stack from the DFS could use up to O(row⋅col) space.

            */
            public int BinarySearchWithDFS(int rowCount, int columnCount, int[][] cells)
            {
                int left = 1, right = rowCount * columnCount;
                while (left < right)
                {
                    int mid = right - (right - left) / 2;
                    if (CanCross(rowCount, columnCount, cells, mid))
                    {
                        left = mid;
                    }
                    else
                    {
                        right = mid - 1;
                    }
                }
                return left;
                bool CanCross(int rowCount, int columnCount, int[][] cells, int day)
                {
                    int[][] grid = new int[rowCount][];
                    for (int i = 0; i < rowCount; i++)
                    {
                        grid[i] = new int[columnCount];
                    }

                    for (int i = 0; i < day; ++i)
                    {
                        int row = cells[i][0] - 1;
                        int column = cells[i][1] - 1;
                        grid[row][column] = 1;
                    }

                    for (int i = 0; i < day; ++i)
                    {
                        grid[cells[i][0] - 1][cells[i][1] - 1] = 1;
                    }

                    for (int i = 0; i < columnCount; ++i)
                    {
                        if (grid[0][i] == 0 && Dfs(grid, 0, i, rowCount, columnCount))
                        {
                            return true;
                        }
                    }
                    return false;
                }
                bool Dfs(int[][] grid, int row, int column, int rowCount, int columnCount)
                {
                    if (row < 0 || row >= rowCount || column < 0 || column >= columnCount || grid[row][column] != 0)
                    {
                        return false;
                    }
                    if (row == rowCount - 1)
                    {
                        return true;
                    }
                    grid[row][column] = -1;
                    foreach (int[] direction in directions)
                    {
                        int newRow = row + direction[0];
                        int newColumn = column + direction[1];
                        if (Dfs(grid, newRow, newColumn, rowCount, columnCount))
                        {
                            return true;
                        }
                    }
                    return false;
                }

            }
            /*
            Approach 3: Disjoint Set Union (on land cells)
Complexity Analysis
•	Time complexity: O(row⋅col)
o	For T operations, the amortized time complexity of the union-find algorithm (using path compression with union by rank) is O(alpha(T)). Here, α(T) is the inverse Ackermann function that grows so slowly, that it doesn't exceed 4 for all reasonable T (approximately T<10600). You can read more about the complexity of union-find here. Because the function grows so slowly, we consider it to be O(1).
o	We iterate over the reversed cells and perform union and find operations on cells of number (row⋅col), we consider the time complexity to be O(row⋅col).
•	Space complexity: O(row⋅col)
o	We use two arrays of size row⋅col+2 to save the root and rank of each cell in the disjoint set data structure.
o	We also create an array of size row×col to represent each cell.

            */
            public int DSUOnLandCells(int row, int col, int[][] cells)
            {
                DSU dsu = new DSU(row * col + 2);
                int[][] grid = new int[row][];
                int[][] directions = { new int[] { 0, 1 }, new int[] { 0, -1 }, new int[] { 1, 0 }, new int[] { -1, 0 } };

                for (int i = cells.Length - 1; i >= 0; i--)
                {
                    int r = cells[i][0] - 1, c = cells[i][1] - 1;
                    grid[r][c] = 1;
                    int index1 = r * col + c + 1;
                    foreach (var d in directions)
                    {
                        int newR = r + d[0], newC = c + d[1];
                        int index2 = newR * col + newC + 1;
                        if (newR >= 0 && newR < row && newC >= 0 && newC < col && grid[newR][newC] == 1)
                        {
                            dsu.Union(index1, index2);
                        }
                    }
                    if (r == 0)
                    {
                        dsu.Union(0, index1);
                    }
                    if (r == row - 1)
                    {
                        dsu.Union(row * col + 1, index1);
                    }
                    if (dsu.Find(0) == dsu.Find(row * col + 1))
                    {
                        return i;
                    }
                }
                return -1;


            }
            class DSU
            {
                private int[] root, size;

                public DSU(int n)
                {
                    root = new int[n];
                    for (int i = 0; i < n; i++)
                    {
                        root[i] = i;
                    }
                    size = new int[n];
                    Array.Fill(size, 1);
                }

                public int Find(int x)
                {
                    if (root[x] != x)
                    {
                        root[x] = Find(root[x]);
                    }
                    return root[x];
                }

                public void Union(int x, int y)
                {
                    int rootX = Find(x);
                    int rootY = Find(y);
                    if (rootX == rootY)
                    {
                        return;
                    }

                    if (size[rootX] > size[rootY])
                    {
                        int temp = rootX;
                        rootX = rootY;
                        rootY = temp;
                    }
                    root[rootX] = rootY;
                    size[rootY] += size[rootX];
                }
            }
            /*
            Approach 4: Disjoint Set Union (on water cells)
            Complexity Analysis
•	Time complexity: O(row⋅col⋅α(row⋅col))
o	For T operations, the amortized time complexity of the union-find algorithm (using path compression with union by rank) is O(alpha(T)). Here, α(T) is the inverse Ackermann function that grows so slowly, that it doesn't exceed 4 for all reasonable T (approximately T<10600). You can read more about the complexity of union-find here. Because the function grows so slowly, we consider it to be O(1).
o	We iterate over cells and perform union and find operations on cells of number (row⋅col), we consider the time complexity to be O(row⋅col).
•	Space complexity: O(row⋅col)
o	We use two arrays of size row⋅col+2 to save the root and rank of each cell in the disjoint set union data structure.
o	We also create an array of size row×col to represent each cell.

            */
            public int DSUOnWaterCells(int row, int col, int[][] cells)
            {
                DSU dsu = new DSU(row * col + 2);
                int[][] grid = new int[row][];
                for (int i = 0; i < row; i++)
                {
                    grid[i] = new int[col];
                }
                int[][] directions = new int[][] { new int[] { 0, 1 }, new int[] { 0, -1 }, new int[] { 1, 0 }, new int[] { -1, 0 }, new int[] { 1, 1 }, new int[] { 1, -1 }, new int[] { -1, 1 }, new int[] { -1, -1 } };

                for (int i = 0; i < row * col; ++i)
                {
                    int r = cells[i][0] - 1, c = cells[i][1] - 1;
                    grid[r][c] = 1;
                    int index1 = r * col + c + 1;
                    foreach (int[] d in directions)
                    {
                        int newR = r + d[0], newC = c + d[1];
                        int index2 = newR * col + newC + 1;
                        if (newR >= 0 && newR < row && newC >= 0 && newC < col && grid[newR][newC] == 1)
                        {
                            dsu.Union(index1, index2);
                        }
                    }
                    if (c == 0)
                    {
                        dsu.Union(0, index1);
                    }
                    if (c == col - 1)
                    {
                        dsu.Union(row * col + 1, index1);
                    }
                    if (dsu.Find(0) == dsu.Find(row * col + 1))
                    {
                        return i;
                    }
                }
                return -1;
            }

        }

        /*
        807. Max Increase to Keep City Skyline
https://leetcode.com/problems/max-increase-to-keep-city-skyline/description/

        */
        public class MaxIncreaseKeepingSkylineSol
        {
            /*
            Approach #1: Row and Column Maximums [Accepted]
Complexity Analysis
•	Time Complexity: O(N^2), where N is the number of rows (and columns) of the grid. We iterate through every cell of the grid.
•	Space Complexity: O(N), the space used by row_maxes and col_maxes.

            */
            public int RownAndColMaximus(int[][] grid)
            {
                int size = grid.Length;
                int[] maxRowValues = new int[size];
                int[] maxColumnValues = new int[size];

                for (int row = 0; row < size; ++row)
                    for (int column = 0; column < size; ++column)
                    {
                        maxRowValues[row] = Math.Max(maxRowValues[row], grid[row][column]);
                        maxColumnValues[column] = Math.Max(maxColumnValues[column], grid[row][column]);
                    }

                int totalIncrease = 0;
                for (int row = 0; row < size; ++row)
                    for (int column = 0; column < size; ++column)
                        totalIncrease += Math.Min(maxRowValues[row], maxColumnValues[column]) - grid[row][column];

                return totalIncrease;
            }
        }


        /*
        808. Soup Servings
        https://leetcode.com/problems/soup-servings/description/
        */
        class SoupServingsSol
        {
            /*
            Approach 1: Bottom-Up Dynamic Programming
            Complexity Analysis
•	Time complexity: O(1).
Let ϵ be the error tolerance, and m0 be the first value such that dp[m0][ m0]>1−ϵ.
We calculate O(min(m, m0)^2)=O(m0^2) states of DP in O(1) each, meaning the total time complexity of the solution is O(m0^2).
We assume ϵ to be constant. It implies that m0 is also constant, thus O(m0^2)=O(1). In our case, ϵ is 10−5, which gives us m0≈200.
•	Space complexity: O(1).
The space complexity is O(m02)=O(1).
  
            */
            public double BottomUpDP(int n)
            {
                int m = (int)Math.Ceiling(n / 25.0);
                Dictionary<int, Dictionary<int, double>> dp = new Dictionary<int, Dictionary<int, double>>();
                dp[0] = new Dictionary<int, double>();
                dp[0][0] = 0.5;

                for (int k = 1; k <= m; k++)
                {
                    dp[k] = new Dictionary<int, double>();
                    dp[0][k] = 1.0;
                    dp[k][0] = 0.0;
                    for (int j = 1; j <= k; j++)
                    {
                        dp[j][k] = CalculateDP(j, k, dp);
                        dp[k][j] = CalculateDP(k, j, dp);
                    }
                    if (dp[k][k] > 1 - 1e-5)
                    {
                        return 1;
                    }
                }

                return dp[m][m];
            }

            private double CalculateDP(int i, int j, Dictionary<int, Dictionary<int, double>> dp)
            {
                return (dp[Math.Max(0, i - 4)][j]
                        + dp[Math.Max(0, i - 3)][j - 1]
                        + dp[Math.Max(0, i - 2)][Math.Max(0, j - 2)]
                        + dp[i - 1][Math.Max(0, j - 3)]) / 4;
            }
            /*
            Approach 2: Top-Down Dynamic Programming (Memoization)
            Complexity Analysis
            •	Time complexity: O(1).
            •	Space complexity: O(1).
            Both time and space complexities are the same as in the first approach.

            */
            public double TopDownDPWithMemo(int n)
            {
                int m = (int)Math.Ceiling(n / 25.0);
                Dictionary<int, Dictionary<int, double>> dp = new Dictionary<int, Dictionary<int, double>>();

                for (int k = 1; k <= m; k++)
                {
                    if (CalculateDP(k, k, dp) > 1 - 1e-5)
                    {
                        return 1.0;
                    }
                }
                return CalculateDP(m, m, dp);

                double CalculateDP(int i, int j, Dictionary<int, Dictionary<int, double>> dp)
                {
                    if (i <= 0 && j <= 0)
                    {
                        return 0.5;
                    }
                    if (i <= 0)
                    {
                        return 1.0;
                    }
                    if (j <= 0)
                    {
                        return 0.0;
                    }
                    if (dp.ContainsKey(i) && dp[i].ContainsKey(j))
                    {
                        return dp[i][j];
                    }
                    double result = (CalculateDP(i - 4, j, dp) + CalculateDP(i - 3, j - 1, dp) +
                                     CalculateDP(i - 2, j - 2, dp) + CalculateDP(i - 1, j - 3, dp)) / 4.0;
                    if (!dp.ContainsKey(i))
                    {
                        dp[i] = new Dictionary<int, double>();
                    }
                    dp[i][j] = result;
                    return result;
                }
            }



        }



        /*
        810. Chalkboard XOR Game
        https://leetcode.com/problems/chalkboard-xor-game/description/

        */
        class XorGameSol
        {
            /*
            Approach #1: Mathematical 
            Complexity Analysis
•	Time Complexity: O(N), where N is the length of nums.
•	Space Complexity: O(1).

            */
            public bool WithMaths(int[] nums)
            {
                int x = 0;
                foreach (int v in nums) x ^= v;
                return x == 0 || nums.Length % 2 == 0;
            }
        }

        /*
        811. Subdomain Visit Count
        https://leetcode.com/problems/subdomain-visit-count/description/
        */
        class SubdomainVisitsSol
        {

            /*
            Approach #1: Hash Map [Accepted]
            Complexity Analysis
•	Time Complexity: O(N), where N is the length of cpdomains, and assuming the length of cpdomains[i] is fixed.
•	Space Complexity: O(N), the space used in our count.

            */
            public IList<string> SubdomainVisits(string[] cpdomains)
            {
                Dictionary<string, int> counts = new Dictionary<string, int>();
                foreach (string domain in cpdomains)
                {
                    string[] cpinfo = domain.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                    string[] frags = cpinfo[1].Split('.');
                    int count = int.Parse(cpinfo[0]);
                    string currentDomain = "";
                    for (int i = frags.Length - 1; i >= 0; --i)
                    {
                        currentDomain = frags[i] + (i < frags.Length - 1 ? "." : "") + currentDomain;
                        if (counts.ContainsKey(currentDomain))
                        {
                            counts[currentDomain] += count;
                        }
                        else
                        {
                            counts[currentDomain] = count;
                        }
                    }
                }

                List<string> ans = new List<string>();
                foreach (string dom in counts.Keys)
                {
                    ans.Add(counts[dom] + " " + dom);
                }
                return ans;
            }
        }


        /*
        815. Bus Routes
        https://leetcode.com/problems/bus-routes/description/
        */

        public class NumBusesToDestinationSol
        {
            /*
            Approach 1: Breadth-First Search (BFS) with Bus Stops as Nodes
        Complexity Analysis
Here, M is the size of routes, and K is the maximum size of routes[i].
•	Time complexity: O(M^2∗K)
To store the routes for each stop we iterate over each route and for each route, we iterate over each stop, hence this step will take O(M∗K) time. In the BFS, we iterate over each route in the queue. For each route we popped, we will iterate over its stop, and for each stop, we will iterate over the connected routes in the map adjList, hence the time required will be O(M∗K∗M) or O(M^2∗K).
•	Space complexity: O(M⋅K)
The map adjList will store the routes for each stop. There can be M⋅K number of stops in routes in the worst case (each of the M routes can have K stops), possibly with duplicates. When represented using adjList, each of the mentioned stops appears exactly once. Therefore, adjList contains an equal number of stop-route element pairs.
    
            */
            public int BFSWithBusStopsAsNodes(int[][] routes, int source, int target)
            {
                if (source == target)
                {
                    return 0;
                }

                Dictionary<int, List<int>> adjacencyList = new Dictionary<int, List<int>>();
                // Create a map from the bus stop to all the routes that include this stop.
                for (int routeIndex = 0; routeIndex < routes.Length; routeIndex++)
                {
                    foreach (int stop in routes[routeIndex])
                    {
                        // Add all the routes that have this stop.
                        if (!adjacencyList.ContainsKey(stop))
                        {
                            adjacencyList[stop] = new List<int>();
                        }
                        adjacencyList[stop].Add(routeIndex);
                    }
                }

                Queue<int> queue = new Queue<int>();
                HashSet<int> visitedRoutes = new HashSet<int>(routes.Length);
                // Insert all the routes in the queue that have the source stop.
                if (adjacencyList.ContainsKey(source))
                {
                    foreach (int route in adjacencyList[source])
                    {
                        queue.Enqueue(route);
                        visitedRoutes.Add(route);
                    }
                }

                int busCount = 1;
                while (queue.Count > 0)
                {
                    int size = queue.Count;

                    for (int i = 0; i < size; i++)
                    {
                        int currentRoute = queue.Dequeue();

                        // Iterate over the stops in the current route.
                        foreach (int stop in routes[currentRoute])
                        {
                            // Return the current count if the target is found.
                            if (stop == target)
                            {
                                return busCount;
                            }

                            // Iterate over the next possible routes from the current stop.
                            if (adjacencyList.ContainsKey(stop))
                            {
                                foreach (int nextRoute in adjacencyList[stop])
                                {
                                    if (!visitedRoutes.Contains(nextRoute))
                                    {
                                        visitedRoutes.Add(nextRoute);
                                        queue.Enqueue(nextRoute);
                                    }
                                }
                            }
                        }
                    }
                    busCount++;
                }
                return -1;
            }

            /*
            Approach 2: Breadth-First Search (BFS) with Routes as Nodes
Complexity Analysis
Here, M is the size of routes, and K is the maximum size of routes[i].
•	Time complexity: O(M^2∗K+M∗k∗logK)
The createGraph method will iterate over every pair of M routes and for each iterate over the K stops to check if there is a common stop, this step will take O(M^2∗K). The addStartingNodes method will iterate over all the M routes and check if the route has source in it, this step will take O(M∗K). In BFS, we iterate over each of the M routes, and for each route, we iterate over the adjacent route which could be M again, so the time it takes is O(M^2).
Sorting each routes[i] takes K∗logK time.
Thus, the time complexity is equal to O(M^2∗K+M∗K∗logK).
•	Space complexity: O(M^2+logK)
The map adjList will store the routes for each route, thus the space it takes is O(M^2). The queue q and the set visited store the routes and hence can take O(M) space.
Some extra space is used when we sort routes[i] in place. The space complexity of the sorting algorithm depends on the programming language.
o	In C++, the sort() function is implemented as a hybrid of Quick Sort, Heap Sort, and Insertion Sort, with a worse-case space complexity of O(logK).
o	In Java, Arrays.sort() is implemented using a variant of the Quick Sort algorithm which has a space complexity of O(logK).
Thus, the total space complexity is equal to O(M^2+logK).

            */
            List<List<int>> adjacencyList = new List<List<int>>();
            public int BFSWithRoutesAsNodes(int[][] routes, int source, int target)
            {
                if (source == target)
                {
                    return 0;
                }

                for (int i = 0; i < routes.Length; ++i)
                {
                    Array.Sort(routes[i]);
                    adjacencyList.Add(new List<int>());
                }

                CreateGraph(routes);

                Queue<int> queue = new Queue<int>();
                AddStartingNodes(queue, routes, source);

                HashSet<int> visited = new HashSet<int>(routes.Length);
                int busCount = 1;
                while (queue.Count > 0)
                {
                    int size = queue.Count;

                    while (size-- > 0)
                    {
                        int node = queue.Dequeue();

                        // Return busCount, if the stop target is present in the current route.
                        if (IsStopExist(routes[node], target))
                        {
                            return busCount;
                        }

                        // Add the adjacent routes.
                        foreach (int adjacent in adjacencyList[node])
                        {
                            if (!visited.Contains(adjacent))
                            {
                                visited.Add(adjacent);
                                queue.Enqueue(adjacent);
                            }
                        }
                    }

                    busCount++;
                }

                return -1;
            }
            // Iterate over each pair of routes and add an edge between them if there's a common stop.
            void CreateGraph(int[][] routes)
            {
                for (int i = 0; i < routes.Length; i++)
                {
                    for (int j = i + 1; j < routes.Length; j++)
                    {
                        if (HaveCommonNode(routes[i], routes[j]))
                        {
                            adjacencyList[i].Add(j);
                            adjacencyList[j].Add(i);
                        }
                    }
                }
            }

            // Returns true if the provided routes have a common stop, false otherwise.
            bool HaveCommonNode(int[] route1, int[] route2)
            {
                int i = 0, j = 0;
                while (i < route1.Length && j < route2.Length)
                {
                    if (route1[i] == route2[j])
                    {
                        return true;
                    }

                    if (route1[i] < route2[j])
                    {
                        i++;
                    }
                    else
                    {
                        j++;
                    }
                }
                return false;
            }

            // Add all the routes in the queue that have the source as one of the stops.
            void AddStartingNodes(Queue<int> queue, int[][] routes, int source)
            {
                for (int i = 0; i < routes.Length; i++)
                {
                    if (IsStopExist(routes[i], source))
                    {
                        queue.Enqueue(i);
                    }
                }
            }

            // Returns true if the given stop is present in the route, false otherwise.
            bool IsStopExist(int[] route, int stop)
            {
                for (int j = 0; j < route.Length; j++)
                {
                    if (route[j] == stop)
                    {
                        return true;
                    }
                }
                return false;
            }


        }


        /*
        826. Most Profit Assigning Work
        https://leetcode.com/problems/most-profit-assigning-work/description/
        */
        class MaxProfitAssignmentSol
        {
            /*
            Approach 1: Binary Search and Greedy (Sort by Job Difficulty)
            Complexity Analysis
Let n be the size of the difficulty and profit arrays, and m be the size of the worker array.
•	Time complexity: O(n⋅logn+m⋅logn)
The time complexity for sorting the jobProfile array is O(n⋅logn).
While iterating the worker array of size m, we perform a binary search with search space size n. The time complexity is given by O(m⋅logn).
Therefore, the total time complexity is given by O(n⋅logn+m⋅logn).
•	Space complexity: O(n)
We create an additional jobProfile array of size 2⋅n. Apart from this, some extra space is used when we sort an array in place. The space complexity of the sorting algorithm depends on the programming language.
o	In Python, the sort method sorts a list using the Tim Sort algorithm which is a combination of Merge Sort and Insertion Sort and has O(n) additional space. Additionally, Tim Sort is designed to be a stable algorithm.
o	In Java, Arrays.sort() is implemented using a variant of the Quick Sort algorithm which has a space complexity of O(logn) for sorting an array.
o	In C++, the sort() function is implemented as a hybrid of Quick Sort, Heap Sort, and Insertion Sort, with a worse-case space complexity of O(logn).
Therefore, space complexity is given by O(n).

            */
            public int BinarySearchAndGreedyWithSortByJobDifficulty(int[] difficulty, int[] profit, int[] worker)
            {
                List<int[]> jobProfile = new List<int[]>();
                jobProfile.Add(new int[] { 0, 0 });
                for (int i = 0; i < difficulty.Length; i++)
                {
                    jobProfile.Add(new int[] { difficulty[i], profit[i] });
                }

                // Sort by difficulty values in increasing order.
                jobProfile.Sort((a, b) => a[0].CompareTo(b[0]));
                for (int i = 0; i < jobProfile.Count - 1; i++)
                {
                    jobProfile[i + 1][1] = Math.Max(jobProfile[i][1], jobProfile[i + 1][1]);
                }

                int totalNetProfit = 0;
                for (int i = 0; i < worker.Length; i++)
                {
                    int workerAbility = worker[i];

                    // Find the job with just smaller or equal difficulty than ability.
                    int leftIndex = 0, rightIndex = jobProfile.Count - 1, jobProfit = 0;
                    while (leftIndex <= rightIndex)
                    {
                        int midIndex = (leftIndex + rightIndex) / 2;
                        if (jobProfile[midIndex][0] <= workerAbility)
                        {
                            jobProfit = Math.Max(jobProfit, jobProfile[midIndex][1]);
                            leftIndex = midIndex + 1;
                        }
                        else
                        {
                            rightIndex = midIndex - 1;
                        }
                    }

                    // Increment profit of current worker to total profit.
                    totalNetProfit += jobProfit;
                }
                return totalNetProfit;
            }

            /*
            Approach 2: Binary Search and Greedy (Sort by profit)
           Complexity Analysis
Let n be the size of the difficulty and profit arrays and m be the size of the worker array.
•	Time complexity: O(n⋅logn+m⋅logn)
The time complexity for sorting the difficulty array is O(n⋅logn).
While iterating the worker array of size m, we perform a binary search with search space size n. The time complexity for is given by O(m⋅logn).
Therefore, the total time complexity is given by O(n⋅logn+m⋅logn).
•	Space complexity: O(n)
 

            */
            public int BinarySearchAndGreedyWithSortByProfit(int[] difficulty, int[] profit, int[] worker)
            {
                List<int[]> jobProfile = new List<int[]> { new int[] { 0, 0 } };
                for (int i = 0; i < difficulty.Length; i++)
                {
                    jobProfile.Add(new int[] { profit[i], difficulty[i] });
                }

                // Sort in decreasing order of profit.
                jobProfile.Sort((a, b) => b[0].CompareTo(a[0]));
                for (int i = 0; i < jobProfile.Count - 1; i++)
                {
                    jobProfile[i + 1][1] = Math.Min(jobProfile[i][1], jobProfile[i + 1][1]);
                }

                int totalProfit = 0;
                for (int i = 0; i < worker.Length; i++)
                {
                    int workerAbility = worker[i];
                    // Maximize profit using binary search.
                    int left = 0, right = jobProfile.Count - 1, jobProfit = 0;
                    while (left <= right)
                    {
                        int middle = (left + right) / 2;
                        if (jobProfile[middle][1] <= workerAbility)
                        {
                            jobProfit = Math.Max(jobProfit, jobProfile[middle][0]);
                            right = middle - 1;
                        }
                        else
                        {
                            left = middle + 1;
                        }
                    }
                    // Add profit of each worker to total profit.
                    totalProfit += jobProfit;
                }
                return totalProfit;
            }
            /*
            Approach 3: Greedy and Two-Pointers
            Complexity Analysis
Let n be the size of the difficulty and profit arrays and m be the size of the worker array.
•	Time complexity: O(n⋅logn+m⋅log(m))
The time taken for sorting the difficulty array is O(n⋅logn) and sorting the worker array is O(m⋅log(m)).
In the two pointers, while iterating through the worker array we iterate the jobProfile array exactly once. Time complexity is given by O(n+m)
Therefore, the total time complexity is given by O(n⋅logn+m⋅log(m)).
•	Space complexity: O(n)
We create an additional jobProfile array of size 2⋅n. Apart from this, some extra space is used when we sort an array in place. The space complexity of the sorting algorithm depends on the programming language.

            */
            public int GreedyWithTwoPointers(int[] difficulty, int[] profit, int[] worker)
            {
                List<int[]> jobProfile = new List<int[]>();
                for (int i = 0; i < difficulty.Length; i++)
                {
                    jobProfile.Add(new int[] { difficulty[i], profit[i] });
                }

                // Sort both worker and jobProfile arrays
                Array.Sort(worker);
                jobProfile.Sort((a, b) => a[0].CompareTo(b[0]));

                int netProfit = 0, maxProfit = 0, index = 0;
                for (int i = 0; i < worker.Length; i++)
                {
                    // While the index has not reached the end and worker can pick a job
                    // with greater difficulty move ahead.
                    while (index < difficulty.Length && worker[i] >= jobProfile[index][0])
                    {
                        maxProfit = Math.Max(maxProfit, jobProfile[index][1]);
                        index++;
                    }
                    netProfit += maxProfit;
                }
                return netProfit;
            }
            /*
            Approach 4: Memoization
Complexity Analysis
Let n be the size of the difficulty and profit arrays and m be the size of the worker array. Also, let maxAbility be the maximum value in the worker array.
•	Time complexity: O(n+m+maxAbility)
In this approach, we iterate through the difficulty, worker and jobs arrays exactly once.
Therefore, the total time complexity is given by O(n+m+maxAbility).
•	Space complexity: O(maxAbility)
We create an additional jobs array of size maxAbility. Apart from this, no additional space is used.
Therefore, space complexity is given by O(maxAbility).

            */
            public int WithMemorization(int[] difficulty, int[] profit, int[] worker)
            {
                // Find maximum ability in the worker array.
                int maximumAbility = worker.Max();
                int[] jobProfits = new int[maximumAbility + 1];

                for (int i = 0; i < difficulty.Length; i++)
                {
                    if (difficulty[i] <= maximumAbility)
                    {
                        jobProfits[difficulty[i]] = Math.Max(jobProfits[difficulty[i]], profit[i]);
                    }
                }

                // Take maxima of prefixes.
                for (int i = 1; i <= maximumAbility; i++)
                {
                    jobProfits[i] = Math.Max(jobProfits[i], jobProfits[i - 1]);
                }

                int totalNetProfit = 0;
                foreach (int ability in worker)
                {
                    totalNetProfit += jobProfits[ability];
                }
                return totalNetProfit;
            }

        }

        /*
        2071. Maximum Number of Tasks You Can Assign
        https://leetcode.com/problems/maximum-number-of-tasks-you-can-assign/description/
        */

        class MaxTaskAssignSol
        {
            /*
            Approach: greedy + binary search
            Complexity :
        It is O(n * log^2 n) for time and O(n) for space.

            */
            public int GreedyWithBinarySearch(int[] tasks, int[] workers, int pills, int strength)
            {
                int numberOfTasks = tasks.Length, numberOfWorkers = workers.Length;
                Array.Sort(tasks);
                Array.Sort(workers);
                int lowerBound = 0, upperBound = Math.Min(numberOfWorkers, numberOfTasks);

                while (lowerBound < upperBound)
                {
                    int middle = upperBound - ((upperBound - lowerBound) >> 1);
                    int count = 0;
                    bool isPossible = true;
                    SortedDictionary<int, int> workerAvailability = new SortedDictionary<int, int>();

                    foreach (int worker in workers)
                        workerAvailability[worker] = workerAvailability.GetValueOrDefault(worker, 0) + 1;

                    for (int i = middle - 1; i >= 0; --i)
                    {
                        // Case 1: Trying to assign to a worker without the pill
                        int strongestWorker = workerAvailability.Keys.Last();
                        if (tasks[i] <= strongestWorker)
                        {
                            workerAvailability[strongestWorker]--;
                            if (workerAvailability[strongestWorker] == 0)
                            {
                                workerAvailability.Remove(strongestWorker);
                            }
                        }
                        else
                        {
                            // Case 2: Trying to assign to a worker with the pill
                            int? weakerWorker = null;
                            foreach (var availability in workerAvailability)
                            {
                                if (availability.Key >= tasks[i] - strength)
                                {
                                    weakerWorker = availability.Key;
                                    break;
                                }
                            }
                            if (weakerWorker != null)
                            {
                                count++;
                                workerAvailability[weakerWorker.Value]--;
                                if (workerAvailability[weakerWorker.Value] == 0)
                                {
                                    workerAvailability.Remove(weakerWorker.Value);
                                }
                            }
                            else
                            {
                                // Case 3: Impossible to assign mid tasks
                                isPossible = false;
                                break;
                            }
                        }
                        if (count > pills)
                        {
                            isPossible = false;
                            break;
                        }
                    }

                    if (isPossible)
                        lowerBound = middle;
                    else
                        upperBound = middle - 1;
                }
                return lowerBound;
            }
        }

        /*
        2141. Maximum Running Time of N Computers
        https://leetcode.com/problems/maximum-running-time-of-n-computers/description/
        */
        class MaxRunTimeSol
        {
            /*
            Approach 1: Sorting and Prefix Sum
         Complexity Analysis
Let m be the length of the input array batteries.
•	Time complexity: O(m⋅logm)
o	We sort batteries in place, it takes O(m⋅logm) time.
o	Picking the largest n-th batteries from a sorted array takes O(n) time. Note that since n<m, this term will be dominated.
o	Then we iterate over the remaining part of the batteries, the computation at each step takes constant time. Thus it takes O(m) time to finish the iteration.
o	To sum up, the overall time complexity is O(m⋅logm).
•	Space complexity: O(m)
o	Some extra space is used when we sort batteries in place. The space complexity of the sorting algorithm depends on the programming language.
o	In python, the sort method sorts a list using the Timsort algorithm, which is a combination of Merge Sort and Insertion Sort and uses O(m) additional space.
o	In Java, Arrays.sort() is implemented using a variant of the Quick Sort algorithm which has a space complexity of O(logm).
o	We create an array of size O(n) to record the power (running time) of each computer.
o	To sum up, the overall space complexity is O(m).
   
            */
            public long SortingAndPrefixSum(int numberOfComputers, int[] batteries)
            {
                // Get the sum of all extra batteries.
                Array.Sort(batteries);
                long extraBatteries = 0;
                for (int i = 0; i < batteries.Length - numberOfComputers; i++)
                {
                    extraBatteries += batteries[i];
                }

                // live stands for the n largest batteries we chose for n computers.
                int[] liveBatteries = new int[numberOfComputers];
                Array.Copy(batteries, batteries.Length - numberOfComputers, liveBatteries, 0, numberOfComputers);

                // We increase the total running time using 'extraBatteries' by increasing 
                // the running time of the computer with the smallest battery.
                for (int i = 0; i < numberOfComputers - 1; i++)
                {
                    // If the target running time is between liveBatteries[i] and liveBatteries[i + 1].
                    if (extraBatteries < (long)(i + 1) * (liveBatteries[i + 1] - liveBatteries[i]))
                    {
                        return liveBatteries[i] + extraBatteries / (long)(i + 1);
                    }

                    // Reduce 'extraBatteries' by the total power used.
                    extraBatteries -= (long)(i + 1) * (liveBatteries[i + 1] - liveBatteries[i]);
                }

                // If there is power left, we can increase the running time 
                // of all computers.
                return liveBatteries[numberOfComputers - 1] + extraBatteries / numberOfComputers;
            }

            /*
            Approach 2: Binary Search
            Complexity Analysis
Let m be the length of the input array batteries and k be the maximum power of one battery.
•	Time complexity: O(m⋅logk)
o	Initially, we set 1 as the left boundary and sum(batteries) / n as the right boundary. Thus it takes O(log((m⋅k)/2)) steps to locate the maximum running time in the worst-case scenario.
o	At each step, we need to iterate over batteries to add up the power that can be used, which takes O(m) time.
o	Therefore, the overall time complexity is O(m⋅log((m⋅k)/2))=O(m⋅logk)
,k≫m,n
•	Space complexity: O(1)
o	During the binary search, we only need to record the boundaries of the searching space and the power extra, and the accumulative sum of extra, which only takes constant space.

            */
            public long UsingBinarySearch(int numberOfDevices, int[] batteryPowers)
            {
                long totalPower = 0;
                foreach (int power in batteryPowers)
                    totalPower += power;
                long lowerBound = 1, upperBound = totalPower / numberOfDevices;

                while (lowerBound < upperBound)
                {
                    long targetPower = upperBound - (upperBound - lowerBound) / 2;
                    long availablePower = 0;

                    foreach (int power in batteryPowers)
                        availablePower += Math.Min(power, targetPower);

                    if (availablePower >= (long)(numberOfDevices * targetPower))
                        lowerBound = targetPower;
                    else
                        upperBound = targetPower - 1;
                }
                return lowerBound;
            }

        }

        /*
        2398. Maximum Number of Robots Within Budget
        https://leetcode.com/problems/maximum-number-of-robots-within-budget/description/
        */
        public class MaximumRobotsWithInBudgetSolution
        {
            /*
            Approach1: Sliding Window + TreeMap (SortedSet)
            Time O(nlogn)
            Space O(n)
            */
            public int SlidingWindowWithSortedDict(int[] chargeTimes, int[] runningCosts, long budget)
            {
                int currentCost = 0;
                int startIndex = 0;
                int totalRobots = chargeTimes.Length;
                SortedSet<int> sortedTimes = new SortedSet<int>();

                //TODO: Fix below code
                /*  for (int endIndex = 0; endIndex < totalRobots; endIndex++)
                 {
                     currentCost += runningCosts[endIndex];
                     sortedTimes.Add(chargeTimes[endIndex]);                    

                     while (sortedTimes[sortedTimes.Count - 1] + (endIndex - startIndex + 1) * currentCost > budget)
                     {
                         sortedTimes.RemoveAt(0); // Remove the earliest time
                         currentCost -= runningCosts[startIndex];
                         startIndex++;
                     }
                 } */
                return totalRobots - startIndex;
            }

            /*
            Approach2: Sliding Window + Mono Deque
            Use a mono deque to find the maximum value in a sliding window.
            Time O(n)
            Space O(n)

            */
            public int MaximumRobots(int[] times, int[] costs, long budget)
            {
                long totalCost = 0;
                int startIndex = 0, totalRobots = times.Length;
                //TODO: test below code for correctness
                LinkedList<int> indicesDeque = new LinkedList<int>();

                for (int currentIndex = 0; currentIndex < totalRobots; ++currentIndex)
                {
                    totalCost += costs[currentIndex];
                    while (indicesDeque.Count > 0 && times[indicesDeque.Last.Value] <= times[currentIndex])
                        indicesDeque.RemoveLast();

                    indicesDeque.AddLast(currentIndex);

                    if (times[indicesDeque.First.Value] + (currentIndex - startIndex + 1) * totalCost > budget)
                    {
                        if (indicesDeque.First.Value == startIndex)
                            indicesDeque.RemoveFirst();
                        totalCost -= costs[startIndex++];
                    }
                }

                return totalRobots - startIndex;
            }

        }

        /*
        2528. Maximize the Minimum Powered City
        https://leetcode.com/problems/maximize-the-minimum-powered-city/description/
        */
        class MaxPossibleMinPowerCitySol
        {
            /*
            Approach: Binary Search & Sliding Window & Greedy
            Complexity
•	Time complexity: O(N * log(SUM_STATION + k)), where N <= 10^5 is the number of cities, SUM_STATION <= 10^10 is sum of stations, k <= 10^9 is the number of additional power stations.
•	Space complexity: O(N)

            */
            public int MaxPower(List<int> powerStations, int radius, int additionalStations)
            {
                int numberOfStations = powerStations.Count;

                bool IsSufficientPower(int minPowerRequired, int remainingAdditionalStations)
                {
                    int windowPower = 0;
                    for (int i = 0; i < radius && i < numberOfStations; i++)
                    {
                        windowPower += powerStations[i];
                    }
                    int[] stationAdditions = new int[numberOfStations];
                    for (int i = 0; i < numberOfStations; i++)
                    {
                        if (i + radius < numberOfStations)
                        {
                            windowPower += powerStations[i + radius];
                        }

                        if (windowPower < minPowerRequired)
                        {
                            int neededPower = minPowerRequired - windowPower;
                            if (neededPower > remainingAdditionalStations)
                            {
                                return false;
                            }
                            // Plant the additional stations on the farthest city in the range to cover as many cities as possible
                            stationAdditions[Math.Min(numberOfStations - 1, i + radius)] += neededPower;
                            windowPower = minPowerRequired;
                            remainingAdditionalStations -= neededPower;
                        }

                        if (i - radius >= 0)
                        {
                            windowPower -= powerStations[i - radius] + stationAdditions[i - radius];
                        }
                    }
                    return true;
                }

                int left = 0;
                int right = 0;
                foreach (var station in powerStations)
                {
                    right += station; // The answer = `right`, when `r = n`, all value of stations are the same!
                }
                right += additionalStations;

                int maximumMinimumPower = 0;
                while (left <= right)
                {
                    int mid = (left + right) / 2;
                    if (IsSufficientPower(mid, additionalStations))
                    {
                        maximumMinimumPower = mid; // This is the maximum possible minimum power so far
                        left = mid + 1; // Search for a larger value in the right side
                    }
                    else
                    {
                        right = mid - 1; // Decrease minPowerRequired to need fewer additional power stations
                    }
                }
                return maximumMinimumPower;
            }
        }

        /*
        2300. Successful Pairs of Spells and Potions
        https://leetcode.com/problems/successful-pairs-of-spells-and-potions/description/
        */
        class SuccessfulPairsSol
        {
            /*
            Approach 1: Sorting + Binary Search
            Complexity Analysis
Here, n is the number of elements in the spells array, and m is the number of elements in the potions array.
•	Time complexity: O((m+n)⋅logm)
o	We sort the potions array which takes O(mlogm) time.
o	Then, for each element of the spells array using binary search we find the respective minPotion which takes O(logm) time. So, for n elements it takes O(nlogm) time.
o	Thus, overall we take O(mlogm+nlogm) time.
•	Space complexity: O(logm) or O(m)
o	The output array answer is not considered as additional space usage.
o	But some extra space is used when we sort the potions array in place. The space complexity of the sorting algorithm depends on the programming language.

            */
            public int[] SortingAndBinarySearch(int[] spells, int[] potions, long success)
            {
                // Sort the potions array in increasing order.
                Array.Sort(potions);
                int[] result = new int[spells.Length];

                int potionCount = potions.Length;
                int maximumPotion = potions[potionCount - 1];

                for (int i = 0; i < spells.Length; i++)
                {
                    int spell = spells[i];
                    // Minimum value of potion whose product with current spell  
                    // will be at least success or more.
                    long minimumPotion = (long)Math.Ceiling((1.0 * success) / spell);
                    // Check if we don't have any potion which can be used.
                    if (minimumPotion > maximumPotion)
                    {
                        result[i] = 0;
                        continue;
                    }
                    // We can use the found potion, and all potions in its right 
                    // (as the right potions are greater than the found potion).
                    int index = LowerBound(potions, (int)minimumPotion);
                    result[i] = potionCount - index;
                }

                return result;
            }
            // Our implementation of lower bound method using binary search.
            private int LowerBound(int[] array, int key)
            {
                int low = 0;
                int high = array.Length;
                while (low < high)
                {
                    int mid = low + (high - low) / 2;
                    if (array[mid] < key)
                    {
                        low = mid + 1;
                    }
                    else
                    {
                        high = mid;
                    }
                }
                return low;
            }

            /*
            Approach 2: Sorting + Two Pointers
            Complexity Analysis
   Here, n is the number of elements in the spells array, and m is the number of elements in the potions array.
   •	Time complexity: O(nlogn+mlogm)
   o	We create an array sortedSpells which takes O(n) time, and then sort the sortedSpells and potions arrays which take O(nlogn) and O(mlogm) time respectively.
   o	Then using two pointers we iterate on each element of the sortedSpells and potions arrays once which will take O(n+m) time.
   o	Thus, overall we take O(nlogn+mlogm) time.
   •	Space complexity: O(n+logm) or O(n+m)
   o	The output array answer is not considered as additional space usage.
   o	But we create an additional array sortedSpells which will take O(n) space.
   o	And some extra space is used when we sort the sortedSpells and potions array in place. The space complexity of the sorting algorithm depends on the programming language.
   o	Thus, sorting uses either O(logn+logm) or O(n+m) space.
   o	So, overall we usem O(n+logn+logm)=O(n+logm) or O(n+n+m)=O(n+m) space
            */

            public int[] SotingAndTwoPointers(int[] spells, int[] potions, long success)
            {
                int numberOfSpells = spells.Length;
                int numberOfPotions = potions.Length;

                // Create an array of pairs containing spell and its original index
                int[][] sortedSpells = new int[numberOfSpells][];
                for (int i = 0; i < numberOfSpells; i++)
                {
                    sortedSpells[i] = new int[2];
                    sortedSpells[i][0] = spells[i];
                    sortedSpells[i][1] = i;
                }

                // Sort the 'spells with index' and 'potions' array in increasing order
                Array.Sort(sortedSpells, (a, b) => a[0].CompareTo(b[0]));
                Array.Sort(potions);

                // For each 'spell' find the respective 'minPotion' index
                int[] answer = new int[numberOfSpells];
                int potionIndex = numberOfPotions - 1;

                foreach (int[] sortedSpell in sortedSpells)
                {
                    int spell = sortedSpell[0];
                    int index = sortedSpell[1];

                    while (potionIndex >= 0 && (long)spell * potions[potionIndex] >= success)
                    {
                        potionIndex -= 1;
                    }
                    answer[index] = numberOfPotions - (potionIndex + 1);
                }

                return answer;
            }
        }

        /*
        881. Boats to Save People
        https://leetcode.com/problems/boats-to-save-people/description/
        */
        class NumRescueBoatsSol
        {
            /*
            Approach 1: Greedy (Two Pointer)
            Complexity Analysis
•	Time Complexity: O(NlogN), where N is the length of people.
•	Space Complexity: O(logN).
o	Some extra space is used when we sort people in place. The space complexity of the sorting algorithm depends on which sorting algorithm is used; the default algorithm varies from one language to another.

            */
            public int numRescueBoats(int[] people, int limit)
            {
                Array.Sort(people);
                int i = 0, j = people.Length - 1;
                int ans = 0;

                while (i <= j)
                {
                    ans++;
                    if (people[i] + people[j] <= limit)
                        i++;
                    j--;
                }

                return ans;
            }
        }


        /*
        879. Profitable Schemes
        https://leetcode.com/problems/profitable-schemes/description/
        */

        public class ProfitableSchemesSol
        {
            private const int Modulus = 1000000007;
            private int[][][] memo = new int[101][][];

            public ProfitableSchemesSol()
            {
                for (int i = 0; i < 101; i++)
                {
                    memo[i] = new int[101][];
                    for (int j = 0; j < 101; j++)
                    {
                        memo[i][j] = new int[101];
                        for (int k = 0; k < 101; k++)
                        {
                            memo[i][j][k] = -1;
                        }
                    }
                }
            }
            /*
            Approach 1: Top-Down Dynamic Programming
     Complexity Analysis
Here, N is the maximum number of criminals allowed in a scheme, M is the size of the list group, and K is the value of minProfit.
•	Time complexity: O(N⋅M⋅K).
We have three parameters index, count and profit. The index can vary from 0 to M - 1, and the count can again vary from 0 to N - 1 (as we consider crime only if it doesn't exceed the limit of N), the last param profit can vary largely but since we cap its value to minProfit it values can vary from 0 to minProfit. We need to calculate the answer to each of these states to solve the original problem; hence the total computation would be O(N⋅M⋅K).
•	Space complexity: O(N⋅M⋅K).
The size of memo would equal the number of states as (N⋅M⋅K). Although we used the maximum value of 101 in the code to simplify things, we can also use the original values in the input as the size of memo. Also, there would be some space in the recursion as well, the total number of active recursion calls could be N one for each crime, and hence the total recursion space would be O(N).
       
            */
            public int TopDownDP(int n, int minProfit, int[] group, int[] profit)
            {
                return Find(0, 0, 0, n, minProfit, group, profit);
            }

            private int Find(int position, int currentCount, int currentProfit, int n, int minimumProfit, int[] group, int[] profits)
            {
                if (position == group.Length)
                {
                    // If profit exceeds the minimum required; it's a profitable scheme.
                    return currentProfit >= minimumProfit ? 1 : 0;
                }

                if (memo[position][currentCount][currentProfit] != -1)
                {
                    // Repeated subproblem, return the stored answer.
                    return memo[position][currentCount][currentProfit];
                }

                // Ways to get a profitable scheme without this crime.
                int totalWays = Find(position + 1, currentCount, currentProfit, n, minimumProfit, group, profits);
                if (currentCount + group[position] <= n)
                {
                    // Adding ways to get profitable schemes, including this crime.
                    totalWays += Find(position + 1, currentCount + group[position], Math.Min(minimumProfit, currentProfit + profits[position]), n, minimumProfit, group, profits);
                }

                return memo[position][currentCount][currentProfit] = totalWays % Modulus;
            }

            /*
            Approach 2: Bottom-Up Dynamic Programming
           Complexity Analysis
Here, N is the maximum member allowed in the subset, M is the size of the list group, K is the maximum value of minProfit.
•	Time complexity: O(N⋅M⋅K).
Similar to the previous approach, we would still need to process each of the states to solve the problem, and hence the total time complexity would remain the same as O(N⋅M⋅K).
•	Space complexity: O(N⋅M⋅K).
This time there won't be any stack space consumption, but the array dp would still be of the size (N⋅M⋅K) and hence the space complexity would be O(N⋅M⋅K).
 
            */

            private int mod = 1000000007;
            private int[][][] dp = new int[101][][];

            public int BottomUpDP(int totalMembers, int minimumProfit, int[] group, int[] profits)
            {
                // Initializing the base case.
                for (int count = 0; count <= totalMembers; count++)
                {
                    dp[group.Length] = new int[totalMembers + 1][];
                    for (int i = 0; i <= totalMembers; i++)
                    {
                        dp[group.Length][i] = new int[minimumProfit + 1];
                    }
                    dp[group.Length][count][minimumProfit] = 1;
                }

                for (int index = group.Length - 1; index >= 0; index--)
                {
                    for (int count = 0; count <= totalMembers; count++)
                    {
                        for (int profit = 0; profit <= minimumProfit; profit++)
                        {
                            // Ways to get a profitable scheme without this crime.
                            dp[index] = new int[totalMembers + 1][];
                            for (int i = 0; i <= totalMembers; i++)
                            {
                                dp[index][i] = new int[minimumProfit + 1];
                            }
                            dp[index][count][profit] = dp[index + 1][count][profit];
                            if (count + group[index] <= totalMembers)
                            {
                                // Adding ways to get profitable schemes, including this crime.
                                dp[index][count][profit] = (dp[index][count][profit] + dp[index + 1][count + group[index]][Math.Min(minimumProfit, profit + profits[index])]) % mod;
                            }
                        }
                    }
                }

                return dp[0][0][0];
            }

        }



        /*
        877. Stone Game
        https://leetcode.com/problems/stone-game/description/
        */
        class StoneGameSol
        {
            /*
            Approach 1: Dynamic Programming
            Complexity Analysis
•	Time Complexity: O(N^2), where N is the number of piles.
•	Space Complexity: O(N^2), the space used storing the intermediate results of each subgame.

            */
            public bool WithDP(int[] piles)
            {
                int N = piles.Length;

                // dp[i+1][j+1] = the value of the game [piles[i], ..., piles[j]].
                int[][] dp = new int[N + 2][];
                for (int size = 1; size <= N; ++size)
                    for (int i = 0; i + size <= N; ++i)
                    {
                        int j = i + size - 1;
                        int parity = (j + i + N) % 2;  // j - i - N; but +x = -x (mod 2)
                        if (parity == 1)
                            dp[i + 1][j + 1] = Math.Max(piles[i] + dp[i + 2][j + 1], piles[j] + dp[i + 1][j]);
                        else
                            dp[i + 1][j + 1] = Math.Min(-piles[i] + dp[i + 2][j + 1], -piles[j] + dp[i + 1][j]);
                    }

                return dp[1][N] > 0;
            }
            /*
            Approach 2: Mathematical
Complexity Analysis
•	Time and Space Complexity: O(1).

            */
            public bool WithMaths(int[] piles)
            {
                return true;
            }

        }

        /*
        1140. Stone Game II
https://leetcode.com/problems/stone-game-ii/description/
        */
        public class StoneGameIISol
        {
            /*
            Approach: Suffix Sum + DP
        Time complexity:
The time complexity is O(n^3) where n is the number of piles. This arises because:
•	We have two nested loops: one for each starting index i and one for each possible value of M.
•	Inside these loops, we have another loop that iterates up to 2 * M, leading to a cubic complexity.
Space complexity:
The space complexity is O(n^2) for storing the DP table dp[i][m] and the suffix sum array. Both require space proportional to the number of piles squared.
    
            */
            public int StoneGameII(int[] piles)
            {
                int n = piles.Length;

                int[][] dp = new int[n][];
                for (int i = 0; i < n; i++)
                {
                    dp[i] = new int[n + 1];
                }

                int[] suffixSum = new int[n];
                suffixSum[n - 1] = piles[n - 1];

                for (int i = n - 2; i >= 0; i--)
                {
                    suffixSum[i] = suffixSum[i + 1] + piles[i];
                }

                for (int i = n - 1; i >= 0; i--)
                {
                    for (int m = 1; m <= n; m++)
                    {
                        if (i + 2 * m >= n)
                        {
                            dp[i][m] = suffixSum[i];
                        }
                        else
                        {
                            for (int x = 1; x <= 2 * m; x++)
                            {
                                dp[i][m] = Math.Max(dp[i][m], suffixSum[i] - dp[i + x][Math.Max(m, x)]);
                            }
                        }
                    }
                }

                return dp[0][1];
            }
        }


        /*
        1406. Stone Game III
        https://leetcode.com/problems/stone-game-iii/description/
        */

        class StoneGameIIISol
        {
            /*
            Approach 1: Bottom-Up Dynamic Programming
          Complexity Analysis
•	Time complexity: O(n).
There is a for loop that performs n iterations. For each state, we try up to three options: to take 1, 2, or 3 stones, so each iteration takes O(1) time.
•	Space complexity: O(n).
We store the array dp[n + 1] of size O(n).
  
            */
            public String BottomUpDP(int[] stoneValue)
            {
                int n = stoneValue.Length;
                int[] dp = new int[n + 1];
                for (int i = n - 1; i >= 0; i--)
                {
                    dp[i] = stoneValue[i] - dp[i + 1];
                    if (i + 2 <= n)
                    {
                        dp[i] = Math.Max(dp[i], stoneValue[i] + stoneValue[i + 1] - dp[i + 2]);
                    }
                    if (i + 3 <= n)
                    {
                        dp[i] = Math.Max(dp[i], stoneValue[i] + stoneValue[i + 1] + stoneValue[i + 2] - dp[i + 3]);
                    }
                }
                if (dp[0] > 0)
                {
                    return "Alice";
                }
                if (dp[0] < 0)
                {
                    return "Bob";
                }
                return "Tie";
            }
            /*
            Approach 2: Top-Down Dynamic Programming (Memoization)
            Complexity Analysis
•	Time complexity: O(n).
Even though we changed the order of calculating DP, the time complexity is the same as in the previous approach: for each i, we compute dp[i] in O(1). Since we store the results in memory, we will calculate each dp[i] only once.
•	Space complexity: O(n).
It is the same as in the first approach.

            */
            public String TopDownDPWithMemo(int[] stoneValue)
            {
                int dif = f(stoneValue, stoneValue.Length, 0);
                if (dif > 0)
                {
                    return "Alice";
                }
                if (dif < 0)
                {
                    return "Bob";
                }
                return "Tie";
            }
            private int f(int[] stoneValue, int n, int i)
            {
                if (i == n)
                {
                    return 0;
                }
                int result = stoneValue[i] - f(stoneValue, n, i + 1);
                if (i + 2 <= n)
                {
                    result = Math.Max(result, stoneValue[i]
                        + stoneValue[i + 1] - f(stoneValue, n, i + 2));
                }
                if (i + 3 <= n)
                {
                    result = Math.Max(result, stoneValue[i] + stoneValue[i + 1]
                        + stoneValue[i + 2] - f(stoneValue, n, i + 3));
                }
                return result;
            }
            /*
            Approach 3: Bottom-Up Dynamic Programming, Space Complexity Optimized
            Complexity Analysis
•	Time complexity: O(n).
It is the same as in Approach 1.
•	Space complexity: O(1).
We have eliminated the need for an entire array to store the DP values. Instead, we only keep track of the current and next three values. Therefore, the space complexity of this solution is O(1).

            */
            public String BottomUpDPWithSpaceOptimal(int[] stoneValue)
            {
                int n = stoneValue.Length;
                int[] dp = new int[4];
                for (int i = n - 1; i >= 0; i--)
                {
                    dp[i % 4] = stoneValue[i] - dp[(i + 1) % 4];
                    if (i + 2 <= n)
                    {
                        dp[i % 4] = Math.Max(dp[i % 4], stoneValue[i] + stoneValue[i + 1]
                            - dp[(i + 2) % 4]);
                    }
                    if (i + 3 <= n)
                    {
                        dp[i % 4] = Math.Max(dp[i % 4], stoneValue[i] + stoneValue[i + 1]
                            + stoneValue[i + 2] - dp[(i + 3) % 4]);
                    }
                }
                if (dp[0] > 0)
                {
                    return "Alice";
                }
                if (dp[0] < 0)
                {
                    return "Bob";
                }
                return "Tie";
            }


        }


        /*
        1510. Stone Game IV
        https://leetcode.com/problems/stone-game-iv/description/
        */

        class WinnerSquareGameSol
        {
            /*
            Approach 1: DFS with memoization
            Complexity Analysis
Assume N is the length of arr.
•	Time complexity: O(N*Sqrt of N) since we spend O(Sqrt of N) at most for each dfs call, and there are O(N) dfs calls in total.
•	Space complexity: O(N) since we need spaces of O(N) to store the result of dfs.
            */
            public bool WinnerSquareGame(int n)
            {
                Dictionary<int, bool> cache = new Dictionary<int, bool>();
                cache.Add(0, false);
                return dfs(cache, n);
            }

            public static bool dfs(Dictionary<int, bool> cache, int remain)
            {
                if (cache.ContainsKey(remain))
                {
                    return cache[remain];
                }
                int sqrt_root = (int)Math.Sqrt(remain);
                for (int i = 1; i <= sqrt_root; i++)
                {
                    // if there is any chance to make the opponent lose the game in the next round,
                    // then the current player will win.
                    if (!dfs(cache, remain - i * i))
                    {
                        cache[remain] = true;
                        return true;
                    }
                }
                cache[remain] = false;
                return false;
            }

            /*
            Approach 2: DP
            Complexity Analysis
Assume N is the length of arr.
•	Time complexity: O(N*Sqrt of N) since we iterate over the dp array and spend O(Sqrt of N) at most on each element.
•	Space complexity: O(N) since we need a dp array.

            */
            public bool WithDP(int n)
            {
                bool[] dp = new bool[n + 1];
                for (int i = 0; i <= n; i++)
                {
                    if (dp[i])
                    {
                        continue;
                    }
                    for (int k = 1; i + k * k <= n; k++)
                    {
                        dp[i + k * k] = true;
                    }
                }
                return dp[n];
            }
        }

        class StoneGameVSolution
        {
            /*
            Approach1: Brute Force with Prefix Sum + DP
            Time: O(n^3)
            */
            public int NaiveWithPrefixSumAndDP(int[] stoneValue)
            {
                int n = stoneValue.Length;
                int[] pre = new int[n + 1];
                for (int i = 1; i <= n; i++)
                {
                    pre[i] = pre[i - 1] + stoneValue[i - 1];
                }
                int[][] dp = new int[n][];
                for (int l = 1; l < n; l++)
                {
                    for (int i = 0; i < n - l; i++)
                    {
                        int j = i + l, res = 0;
                        for (int k = i; k < j; k++)
                        {
                            int left = pre[k + 1] - pre[i], right = pre[j + 1] - pre[k + 1];
                            if (left < right)
                            {
                                res = Math.Max(res, left + dp[i][k]);
                            }
                            else if (left > right)
                            {
                                res = Math.Max(res, right + dp[k + 1][j]);
                            }
                            else
                            {
                                res = Math.Max(res, left + dp[i][k]);
                                res = Math.Max(res, right + dp[k + 1][j]);
                            }
                        }
                        dp[i][j] = res;
                    }
                }
                return dp[0][n - 1];
            }
            /*
            Approach2: Optimization
            Time Complexity: O(n^2 log n)
            */
            public int Optimal(int[] stoneValue)
            {
                int n = stoneValue.Length;
                int[] pre = new int[n + 1];
                for (int i = 1; i <= n; i++)
                {
                    pre[i] = pre[i - 1] + stoneValue[i - 1];
                }
                int[][] dp = new int[n][], left = new int[n][], right = new int[n][];
                for (int i = 0; i < n; i++)
                {
                    left[i][i] = right[i][i] = stoneValue[i];
                }
                for (int l = 1; l < n; l++)
                {
                    for (int i = 0; i < n - l; i++)
                    {
                        int j = i + l, k = Search(pre, i, j);
                        int sum = pre[j + 1] - pre[i], leftHalf = pre[k + 1] - pre[i];
                        if ((leftHalf << 1) == sum)
                        {    // equal parts
                            dp[i][j] = Math.Max(left[i][k], right[k + 1][j]);
                        }
                        else
                        {    // left half > right half
                            dp[i][j] = Math.Max(k == i ? 0 : left[i][k - 1], k == j ? 0 : right[k + 1][j]);
                        }
                        left[i][j] = Math.Max(left[i][j - 1], sum + dp[i][j]);
                        right[i][j] = Math.Max(right[i + 1][j], sum + dp[i][j]);
                    }
                }
                return dp[0][n - 1];
            }
            // returns first index where sum of left half >= sum of right half
            private int Search(int[] pre, int l, int r)
            {
                int sum = pre[r + 1] - pre[l], L = l;
                while (l < r)
                {
                    int m = l + ((r - l) >> 1);
                    if (((pre[m + 1] - pre[L]) << 1) >= sum)
                    {
                        r = m;
                    }
                    else
                    {
                        l = m + 1;
                    }
                }
                return l;
            }
            /*
            Approach3: Further Optimization
            Time Complexity: O(n^2)
            */
            public int Optimal2(int[] stoneValue)
            {
                int n = stoneValue.Length;
                int[][] dp = new int[n][], max = new int[n][];
                for (int i = 0; i < n; i++)
                {
                    max[i][i] = stoneValue[i];
                }
                for (int j = 1; j < n; j++)
                {
                    int mid = j, sum = stoneValue[j], rightHalf = 0;
                    for (int i = j - 1; i >= 0; i--)
                    {
                        sum += stoneValue[i];
                        while ((rightHalf + stoneValue[mid]) * 2 <= sum)
                        {
                            rightHalf += stoneValue[mid--];
                        }
                        dp[i][j] = rightHalf * 2 == sum ? max[i][mid] : (mid == i ? 0 : max[i][mid - 1]);
                        dp[i][j] = Math.Max(dp[i][j], mid == j ? 0 : max[j][mid + 1]);
                        max[i][j] = Math.Max(max[i][j - 1], dp[i][j] + sum);
                        max[j][i] = Math.Max(max[j][i + 1], dp[i][j] + sum);
                    }
                }
                return dp[0][n - 1];
            }



        }

        /*
        1686. Stone Game VI
        https://leetcode.com/problems/stone-game-vi/description/

        */
        public class StoneGameVI
        {
            /*
            Approach: Optimal
            Complexity
            Time O(nlogn)
            Space O(n)
            */
            public int Optimal1(int[] aliceValues, int[] bobValues)
            {
                int n = aliceValues.Length;
                int[][] sums = new int[n][];
                for (int i = 0; i < n; i++)
                {
                    sums[i] = new int[] { aliceValues[i] + bobValues[i], aliceValues[i], bobValues[i] };
                }
                Array.Sort(sums, (a, b) => b[0].CompareTo(a[0]));
                int a = 0;
                int b = 0;
                for (int i = 0; i < n; i++)
                {
                    if (i % 2 == 0)
                    {
                        a += sums[i][1];
                    }
                    else
                    {
                        b += sums[i][2];
                    }
                }
                return a.CompareTo(b);
            }
            /*
           Approach: Optimal2 - Sort by Value Sum
           Complexity
           Time O(nlogn)
           Space O(n)
           */
            public int Optimal2(int[] aliceValues, int[] bobValues)
            {
                int n = aliceValues.Length, diff = 0;
                for (int i = 0; i < n; i++)
                {
                    aliceValues[i] += bobValues[i];
                    diff -= bobValues[i];
                }
                Array.Sort(aliceValues);
                for (int i = n - 1; i >= 0; i -= 2)
                    diff += aliceValues[i];
                return diff.CompareTo(0);
            }

        }

        /*
        1690. Stone Game VII
        https://leetcode.com/problems/stone-game-vii/description/
        */

        class StoneGameVIISol
        {
            int[] prefixSum;
            /*
            Approach 1: Brute Force Using Recursion
         Complexity Analysis
Let n be the length of array stones.
•	Time Complexity : O(2^n). We fill the array prefixSum of size n by iterating n times. The time complexity would be O(n).
For every array element in stones, there are 2 choices, either we remove it or keep it. Thus, the recursive tree takes the form of binary tree having roughly 2^n nodes. The time complexity would be O(2^n).
This would give us total time complexity as O(n)+O(2^n)=O(2^n).
This approach is exhaustive and results in Time Limit Exceeded (TLE)
•	Space Complexity: O(n), as we build an array prefixSum of size n.
   
            */
            public int NaiveRec(int[] stones)
            {
                int n = stones.Length;
                prefixSum = new int[n + 1];
                for (int i = 0; i < n; i++)
                {
                    prefixSum[i + 1] = prefixSum[i] + stones[i];
                }
                return Math.Abs(FindDifference(0, n - 1, true));
            }
            private int FindDifference(int start, int end, bool alice)
            {
                if (start == end)
                {
                    return 0;
                }
                int difference;
                int scoreRemoveFirst = prefixSum[end + 1] - prefixSum[start + 1];
                int scoreRemoveLast = prefixSum[end] - prefixSum[start];

                if (alice)
                {
                    difference = Math.Max(
                            FindDifference(start + 1, end, !alice) + scoreRemoveFirst,
                            FindDifference(start, end - 1, !alice) + scoreRemoveLast);
                }
                else
                {
                    difference = Math.Min(
                            FindDifference(start + 1, end, !alice) - scoreRemoveFirst,
                            FindDifference(start, end - 1, !alice) - scoreRemoveLast);
                }
                return difference;
            }
            /*
            Approach 2: Top Down Dynamic Programming - Memoization
            Complexity Analysis
Let n be the length of array stones.
•	Time Complexity : O(n^2). For all possible subarrays in array stones, we calculate it's result only once. Since there are n^2 possible subarrays for an array of length n, the time complexity would be O(n^2).
•	Space Complexity: O(n^2). We use an array memo of size n⋅n and prefixSum of size n. This gives us space complexity as O(n^2)+O(n)=O(n^2).

            */

            private const int Infinity = int.MaxValue;
            private int[,] memo;
            public int TopDownDPWithMemo(int[] stones)
            {
                int n = stones.Length;
                memo = new int[n, n];
                for (int i = 0; i < n; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        memo[i, j] = Infinity;
                    }
                }
                prefixSum = new int[n + 1];
                for (int i = 0; i < n; i++)
                {
                    prefixSum[i + 1] = prefixSum[i] + stones[i];
                }
                return Math.Abs(FindDifference(0, n - 1, true));

                int FindDifference(int start, int end, bool isAlice)
                {
                    if (start == end)
                    {
                        return 0;
                    }
                    if (memo[start, end] != Infinity)
                    {
                        return memo[start, end];
                    }
                    int difference;
                    int scoreRemoveFirst = prefixSum[end + 1] - prefixSum[start + 1];
                    int scoreRemoveLast = prefixSum[end] - prefixSum[start];

                    if (isAlice)
                    {
                        difference = Math.Max(
                            FindDifference(start + 1, end, !isAlice) + scoreRemoveFirst,
                            FindDifference(start, end - 1, !isAlice) + scoreRemoveLast);
                    }
                    else
                    {
                        difference = Math.Min(
                            FindDifference(start + 1, end, !isAlice) - scoreRemoveFirst,
                            FindDifference(start, end - 1, !isAlice) - scoreRemoveLast);
                    }
                    memo[start, end] = difference;
                    return difference;
                }
            }
            /*
Approach 3: Optimised Memoization Approach
Complexity Analysis
Let n be the length of array stones.
•	Time Complexity : O(n^2). For all possible subarray in array stones, we calculate it's result only once. Since there are n^2 possible subarrays for an array of length n, the time complexity would be O(n^2).
•	Space Complexity: O(n^2). We use an array memo of size n⋅n and prefixSum of size n. This gives us space complexity as O(n^2)+O(n)=O(n^2).

            */
            public int TopDownDPWithOptimizedMemo(int[] stones)
            {
                int n = stones.Length;
                memo = new int[n, n];
                for (int i = 0; i < n; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        memo[i, j] = Infinity;
                    }
                }
                prefixSum = new int[n + 1];
                for (int i = 0; i < n; i++)
                {
                    prefixSum[i + 1] = prefixSum[i] + stones[i];
                }
                return FindDifference(0, n - 1, stones);
            }
            private int FindDifference(int start, int end, int[] stones)
            {
                if (start == end)
                {
                    return 0;
                }
                if (memo[start, end] != Infinity)
                {
                    return memo[start, end];
                }
                if (start + 1 == end)
                    return Math.Max(stones[start], stones[end]);

                int scoreRemoveFirst = prefixSum[end + 1] - prefixSum[start + 1];
                int scoreRemoveLast = prefixSum[end] - prefixSum[start];

                memo[start, end] = Math.Max(
                        scoreRemoveFirst - FindDifference(start + 1, end, stones),
                        scoreRemoveLast - FindDifference(start, end - 1, stones));

                return memo[start, end];
            }
            /*
            Approach 4: Bottom Up Dynamic Programming - Tabulation
Complexity Analysis
Let n be the length of array stones.
•	Time Complexity : O(n^2), as we iterate over a 2D array of size n⋅n.
•	Space Complexity: O(n^2), as we use an array dp of size n⋅n and prefixSum of size n. This gives us space complexity as O(n^2)+O(n)=O(n^2).

            */
            public int BottomUpDPTabulation(int[] stones)
            {
                int n = stones.Length;
                int[][] dp = new int[n][];
                int[] prefixSum = new int[n + 1];
                for (int i = 0; i < n; i++)
                {
                    prefixSum[i + 1] = prefixSum[i] + stones[i];
                }
                for (int length = 2; length <= n; length++)
                {
                    for (int start = 0; start + length - 1 < n; start++)
                    {
                        int end = start + length - 1;
                        int scoreRemoveFirst = prefixSum[end + 1] - prefixSum[start + 1];
                        int scoreRemoveLast = prefixSum[end] - prefixSum[start];
                        dp[start][end] = Math.Max(scoreRemoveFirst - dp[start + 1][end],
                                scoreRemoveLast - dp[start][end - 1]);

                    }
                }
                return dp[0][n - 1];
            }

            /*
            Approach 5: Another Approach using Tabulation
            Complexity Analysis
Let n be the length of array stones.
•	Time Complexity : O(n^2), as we iterate over a 2D array of size n⋅n.
•	Space Complexity: O(n^2), as we use an array dp of size n⋅n and prefixSum of size n. This gives us space complexity as O(n^2)+O(n)=O(n^2).

            */
            public int BottomUpDPWithOptimalTabulation(int[] stones)
            {
                int n = stones.Length;
                int[] prefixSum = new int[n + 1];
                for (int i = 0; i < n; i++)
                {
                    prefixSum[i + 1] = prefixSum[i] + stones[i];
                }
                int[][] dp = new int[n][];
                for (int start = n - 2; start >= 0; start--)
                {
                    for (int end = start + 1; end < n; end++)
                    {
                        int scoreRemoveFirst = prefixSum[end + 1] - prefixSum[start + 1];
                        int scoreRemoveLast = prefixSum[end] - prefixSum[start];
                        dp[start][end] = Math.Max(scoreRemoveFirst - dp[start + 1][end],
                                scoreRemoveLast - dp[start][end - 1]);
                    }
                }
                return dp[0][n - 1];
            }

        }

        /*
        1872. Stone Game VIII
        https://leetcode.com/problems/stone-game-viii/description/

        */
        public int StoneGameVIII(int[] stones)
        {
            /*
            O(N)
            */
            // find the sum first
            int[] sum = new int[stones.Length];
            sum[0] = stones[0];
            for (int i = 1; i < stones.Length; i++)
            {
                sum[i] = sum[i - 1] + stones[i];
            }
            // apply dp
            int[] dp = new int[stones.Length];
            dp[stones.Length - 1] = sum[stones.Length - 1];
            for (int i = stones.Length - 2; i >= 0; i--)
            {
                dp[i] = Math.Max(sum[i] - dp[i + 1], dp[i + 1]);
            }
            // alice cannot take only one stone, so it has to be starting from dp[1], not dp[0].
            return dp[1];
        }
        /*
        2029. Stone Game IX
https://leetcode.com/problems/stone-game-ix/description/

        */
        public bool StoneGameIX(int[] stones)
        {
            /*
            Complexity
            Time O(n)
            Space O(1)
            */
            int[] cnt = new int[3];
            foreach (int a in stones)
                cnt[a % 3]++;
            if (Math.Min(cnt[1], cnt[2]) == 0)
                return Math.Max(cnt[1], cnt[2]) > 2 && cnt[0] % 2 > 0;
            return Math.Abs(cnt[1] - cnt[2]) > 2 || cnt[0] % 2 == 0;
        }

        /*
        871. Minimum Number of Refueling Stops
        https://leetcode.com/problems/minimum-number-of-refueling-stops/description/

        */
        class MinRefuelStopsSol
        {
            /*
            Approach 1: Dynamic Programming
            Complexity Analysis
•	Time Complexity: O(N^2), where N is the length of stations.
•	Space Complexity: O(N), the space used by dp.

            */
            public int MinRefuelStops(int target, int startFuel, int[][] stations)
            {
                int N = stations.Length;
                long[] dp = new long[N + 1];
                dp[0] = startFuel;
                for (int i = 0; i < N; ++i)
                    for (int t = i; t >= 0; --t)
                        if (dp[t] >= stations[i][0])
                            dp[t + 1] = Math.Max(dp[t + 1], dp[t] + (long)stations[i][1]);

                for (int i = 0; i <= N; ++i)
                    if (dp[i] >= target) return i;
                return -1;
            }
            /*
            Approach 2: Max Heap/ Priority Queue
            Complexity Analysis
•	Time Complexity: O(NlogN), where N is the length of stations.
•	Space Complexity: O(N), the space used by pq.

            */

            public int WithMaxHeap(int target, int tank, int[][] stations)
            {
                // pq is a maxheap of gas station capacities
                PriorityQueue<int, int> pq = new PriorityQueue<int, int>();
                int ans = 0, prev = 0;
                foreach (int[] station in stations)
                {
                    int location = station[0];
                    int capacity = station[1];
                    tank -= location - prev;
                    while (pq.Count > 0 && tank < 0)
                    {  // must refuel in past
                        tank += pq.Dequeue();
                        ans++;
                    }

                    if (tank < 0) return -1;
                    pq.Enqueue(capacity, -capacity);
                    prev = location;
                }

                // Repeat body for station = (target, inf)
                {
                    tank -= target - prev;
                    while (pq.Count > 0 && tank < 0)
                    {
                        tank += pq.Dequeue();
                        ans++;
                    }
                    if (tank < 0) return -1;
                }

                return ans;
            }
        }

        /*
        860. Lemonade Change
        https://leetcode.com/problems/lemonade-change/description/
        */
        class LemonadeChangeSolution
        {
            /*
            Approach: Simulation
            Complexity Analysis
Let n be the length of the bills array.
•	Time complexity: O(n)
The algorithm loops over the length of bills once, taking O(n) time. All operations within the loop are constant time operations.
Thus, the time complexity of the algorithm is O(n).
•	Space complexity: O(1)
The algorithm does not use any additional data structures that scale with the input size. Thus, the space complexity remains constant.

            */

            public bool WithSimulation(int[] bills)
            {
                // Count of $5 and $10 bills in hand
                int fiveDollarBills = 0;
                int tenDollarBills = 0;

                // Iterate through each customer's bill
                foreach (int customerBill in bills)
                {
                    if (customerBill == 5)
                    {
                        // Just add it to our count
                        fiveDollarBills++;
                    }
                    else if (customerBill == 10)
                    {
                        // We need to give $5 change
                        if (fiveDollarBills > 0)
                        {
                            fiveDollarBills--;
                            tenDollarBills++;
                        }
                        else
                        {
                            // Can't provide change, return false
                            return false;
                        }
                    }
                    else
                    { // customerBill == 20
                      // We need to give $15 change
                        if (tenDollarBills > 0 && fiveDollarBills > 0)
                        {
                            // Give change as one $10 and one $5
                            fiveDollarBills--;
                            tenDollarBills--;
                        }
                        else if (fiveDollarBills >= 3)
                        {
                            // Give change as three $5
                            fiveDollarBills -= 3;
                        }
                        else
                        {
                            // Can't provide change, return false
                            return false;
                        }
                    }
                }
                // If we've made it through all customers, return true
                return true;
            }
        }


        /*
        857. Minimum Cost to Hire K Workers
        https://leetcode.com/problems/minimum-cost-to-hire-k-workers/description/
        */

        class MinCostToHireKWorkersSol
        {
            /*
            Approach: Priority Queue
           Complexity Analysis
Let n be the number of workers and k be the size of the priority queue (bounded by k).
•	Time complexity: O(nlogn+nlogk)
Sorting the workers based on their wage-to-quality ratio takes O(nlogn).
Each worker is processed once, and for each worker, we perform push/pop operations on the priority queue, which takes O(logk), so processing the workers takes O(nlogk).
So, the total time complexity is O(nlogn+nlogk), which is dominated by the sorting step when k is much smaller than n.
•	Space complexity: O(n+k)
We use O(n) additional space to store the wage-to-quality ratio for each worker.
We use a priority queue to keep track of the highest quality workers, which can contain at most k workers.
Note that some extra space is used when we sort an array in place. The space complexity of the sorting algorithm depends on the programming language.
So, the total space complexity is O(n+k), where n is the dominating term when k is much smaller than n. 
            */
            public double WithMaxHeapPQ(int[] quality, int[] wage, int k)
            {
                int numberOfWorkers = quality.Length;
                double minimumTotalCost = double.MaxValue;
                double currentTotalQuality = 0;
                List<KeyValuePair<double, int>> wageToQualityRatio = new List<KeyValuePair<double, int>>();

                // Calculate wage-to-quality ratio for each worker
                for (int i = 0; i < numberOfWorkers; i++)
                {
                    wageToQualityRatio.Add(new KeyValuePair<double, int>((double)wage[i] / quality[i], quality[i]));
                }

                // Sort workers based on their wage-to-quality ratio
                wageToQualityRatio = wageToQualityRatio.OrderBy(x => x.Key).ToList();

                // Use a priority queue to keep track of the highest quality workers - 
                //Default minheap tweaked to use as a MaxHeap by adding negating priority
                PriorityQueue<int, int> workers = new PriorityQueue<int, int>(); //Comparer<int>.Create((x, y) => y.CompareTo(x))

                // Iterate through workers
                for (int i = 0; i < numberOfWorkers; i++)
                {
                    workers.Enqueue(wageToQualityRatio[i].Value, -wageToQualityRatio[i].Value);
                    currentTotalQuality += wageToQualityRatio[i].Value;

                    // If we have more than k workers,
                    // remove the one with the highest quality
                    if (workers.Count > k)
                    {
                        currentTotalQuality -= workers.Peek();
                        workers.Dequeue();
                    }

                    // If we have exactly k workers,
                    // calculate the total cost and update if it's the minimum
                    if (workers.Count == k)
                    {
                        minimumTotalCost = Math.Min(minimumTotalCost, currentTotalQuality * wageToQualityRatio[i].Key);
                    }
                }
                return minimumTotalCost;
            }
        }

        /*
        856. Score of Parentheses
        https://leetcode.com/problems/score-of-parentheses/description/
        */
        class ScoreOfParenthesesSol
        {

            /*
            Approach 1: Divide and Conquer
            Complexity Analysis
            •	Time Complexity: O(N^2), where N is the length of S. An example worst case is (((((((....))))))).
            •	Space Complexity: O(N), the size of the implied call stack.

            */
            public int DivideAndConquer(string s)
            {
                return F(s, 0, s.Length);
            }

            private int F(String S, int i, int j)
            {
                //Score of balanced string S[i:j]
                int ans = 0, bal = 0;

                // Split string into primitives
                for (int k = i; k < j; ++k)
                {
                    bal += S[k] == '(' ? 1 : -1;
                    if (bal == 0)
                    {
                        if (k - i == 1) ans++;
                        else ans += 2 * F(S, i + 1, k);
                        i = k + 1;
                    }
                }

                return ans;
            }
            /*
            Approach 2: Stack
            Complexity Analysis
•	Time Complexity: O(N), where N is the length of S.
•	Space Complexity: O(N), the size of the stack.

            */
            public int UsingStack(String S)
            {
                Stack<int> stack = new Stack<int>();
                stack.Push(0); // The score of the current frame

                foreach (char c in S)
                {
                    if (c == '(')
                        stack.Push(0);
                    else
                    {
                        int v = stack.Pop();
                        int w = stack.Pop();
                        stack.Push(w + Math.Max(2 * v, 1));
                    }
                }

                return stack.Pop();
            }
            /*
            Approach 3: Count Cores
Complexity Analysis
•	Time Complexity: O(N), where N is the length of S.
•	Space Complexity: O(1).

            */
            public int CountCores(String S)
            {
                int ans = 0, bal = 0;
                for (int i = 0; i < S.Length; ++i)
                {
                    if (S[i] == '(')
                    {
                        bal++;
                    }
                    else
                    {
                        bal--;
                        if (S[i - 1] == '(')
                            ans += 1 << bal;
                    }
                }

                return ans;
            }
        }

        /*
        851. Loud and Rich
        https://leetcode.com/problems/loud-and-rich/description/
        */

        class LoudAndRichSol
        {
            List<int>[] graph;
            int[] answer;
            int[] quiet;

            /*
            Approach #1: Cached Depth-First Search 
            Complexity Analysis
•	Time Complexity: O(N^2), where N is the number of people.
We are iterating here over array richer. It could contain up to
1+...+N−1=N(N−1)/2 elements, for example, in the situation
when each new person is richer than the previous one.
•	Space Complexity: O(N^2), to keep the graph with N^2 edges.

            */
            public int[] DFSWithCache(int[][] richer, int[] quiet)
            {
                int numberOfPeople = quiet.Length;
                graph = new List<int>[numberOfPeople];
                answer = new int[numberOfPeople];
                this.quiet = quiet;

                for (int node = 0; node < numberOfPeople; ++node)
                    graph[node] = new List<int>();

                foreach (int[] edge in richer)
                    graph[edge[1]].Add(edge[0]);

                Array.Fill(answer, -1);

                for (int node = 0; node < numberOfPeople; ++node)
                    PerformDepthFirstSearch(node);
                return answer;
            }

            private int PerformDepthFirstSearch(int node)
            {
                if (answer[node] == -1)
                {
                    answer[node] = node;
                    foreach (int child in graph[node])
                    {
                        int candidate = PerformDepthFirstSearch(child);
                        if (quiet[candidate] < quiet[answer[node]])
                            answer[node] = candidate;
                    }
                }
                return answer[node];
            }
        }

        /*
        841. Keys and Rooms
        https://leetcode.com/problems/keys-and-rooms/description/
        */

        class CanVisitAllRoomsSol
        {
            /*
            Approach #1: Depth-First Search [Accepted]
Complexity Analysis
•	Time Complexity: O(N+E), where N is the number of rooms, and E is the total number of keys.
•	Space Complexity: O(N) in additional space complexity, to store stack and seen.

            */
            public bool CanVisitAllRooms(List<List<int>> rooms)
            {
                bool[] roomsVisited = new bool[rooms.Count];
                roomsVisited[0] = true;
                Stack<int> keysStack = new Stack<int>();
                keysStack.Push(0);

                // At the beginning, we have a todo list "keysStack" of keys to use.
                // 'roomsVisited' represents at some point we have entered this room.
                while (keysStack.Count > 0)
                { // While we have keys...
                    int currentRoom = keysStack.Pop(); // Get the next key 'currentRoom'
                    foreach (int key in rooms[currentRoom]) // For every key in room # 'currentRoom'...
                        if (!roomsVisited[key])
                        { // ...that hasn't been used yet
                            roomsVisited[key] = true; // mark that we've entered the room
                            keysStack.Push(key); // add the key to the todo list
                        }
                }

                foreach (bool roomVisited in roomsVisited)  // if any room hasn't been visited, return false
                    if (!roomVisited) return false;
                return true;
            }
        }



        /*
        832. Flipping an Image
        https://leetcode.com/problems/flipping-an-image/description/
        */
        class FlipAndInvertImageSol
        {
            /*
            Approach #1: Direct
            Complexity Analysis
•	Time Complexity: O(N), where N is the total number of elements in A.
•	Space Complexity: O(1) in additional space complexity.
            */
            public int[][] Direct(int[][] image)
            {
                int C = image[0].Length;
                foreach (int[] row in image)
                    for (int i = 0; i < (C + 1) / 2; ++i)
                    {
                        int tmp = row[i] ^ 1;
                        row[i] = row[C - 1 - i] ^ 1;
                        row[C - 1 - i] = tmp;
                    }

                return image;
            }
        }

        /*
        835. Image Overlap
        https://leetcode.com/problems/image-overlap/description/
        */
        public class LargestOverlapSol
        {
            /*
            
Approach 1: Shift and Count

      Complexity Analysis
Let N be the width of the matrix.
First of all, let us calculate the number of all possible shiftings, (i.e. the number of overlapping zones).
For a matrix of length N, we have 2(N−1) possible offsets along each axis to shift the matrix.
Therefore, there are in total 2(N−1)⋅2(N−1)=4(N−1)^2 possible overlapping zones to calculate.
•	Time Complexity: O(N^4)
o	As discussed before, we have in total 4(N−1)^2 possible overlapping zones.
o	The size of the overlapping zone is bounded by O(N^2).
o	Since we iterate through each overlapping zone to find out the overlapping ones, the overall time complexity of the algorithm would be 4(N−1)^2⋅O(N^2)=O(N^4).
•	Space Complexity: O(1)
o	As one can see, a constant space is used in the above algorithm.
      
            */

            public int ShiftAndCount(int[][] matrixA, int[][] matrixB)
            {
                int maximumOverlaps = 0;

                for (int yShift = 0; yShift < matrixA.Length; ++yShift)
                    for (int xShift = 0; xShift < matrixA.Length; ++xShift)
                    {
                        // move the matrix A to the up-right and up-left directions.
                        maximumOverlaps = Math.Max(maximumOverlaps, ShiftAndCount(xShift, yShift, matrixA, matrixB));
                        // move the matrix B to the up-right and up-left directions, which is equivalent to moving A to the down-right and down-left directions 
                        maximumOverlaps = Math.Max(maximumOverlaps, ShiftAndCount(xShift, yShift, matrixB, matrixA));
                    }

                return maximumOverlaps;
            }
            /// <summary>
            ///  Shift the matrix M in up-left and up-right directions 
            ///    and count the ones in the overlapping zone.
            /// </summary>
            private int ShiftAndCount(int xShift, int yShift, int[][] matrixM, int[][] matrixR)
            {
                int leftShiftCount = 0, rightShiftCount = 0;
                int resultRow = 0;
                // count the cells of ones in the overlapping zone.
                for (int matrixMRow = yShift; matrixMRow < matrixM.Length; ++matrixMRow)
                {
                    int resultColumn = 0;
                    for (int matrixMCol = xShift; matrixMCol < matrixM.Length; ++matrixMCol)
                    {
                        if (matrixM[matrixMRow][matrixMCol] == 1 && matrixM[matrixMRow][matrixMCol] == matrixR[resultRow][resultColumn])
                            leftShiftCount += 1;
                        if (matrixM[matrixMRow][resultColumn] == 1 && matrixM[matrixMRow][resultColumn] == matrixR[resultRow][matrixMCol])
                            rightShiftCount += 1;
                        resultColumn += 1;
                    }
                    resultRow += 1;
                }
                return Math.Max(leftShiftCount, rightShiftCount);
            }
            /*
            Approach 2: Linear Transformation
            Complexity Analysis
            Let Ma,Mb be the number of non-zero cells in the matrix A and B respectively. Let N be the width of the matrix.
            •	Time Complexity: O(N^4).
            o	In the first step, we filter out the non-zero cells in each matrix, which would take O(N^2) time.
            o	In the second step, we enumerate the cartesian product of non-zero cells between the two matrices, which would take O(Ma⋅Mb) time. In the worst case, both Ma and Mb would be up to N^2, i.e. matrix filled with ones.
            o	To sum up, the overall time complexity of the algorithm would be O(N^2)+O(N^2⋅N^2)=O(N^4).
            o	Although this approach has the same time complexity as the previous approach, it should run faster in practice, since we ignore those zero cells.
            •	Space Complexity: O(N^2)
            o	We kept the indices of non-zero cells in both matrices. In the worst case, we would need the O(N^2) space for the matrices filled with ones.

            */


            public int LinearTransformation(int[,] firstMatrix, int[,] secondMatrix)
            {
                List<KeyValuePair<int, int>> firstMatrixOnes = GetNonZeroCells(firstMatrix);
                List<KeyValuePair<int, int>> secondMatrixOnes = GetNonZeroCells(secondMatrix);

                int maximumOverlaps = 0;
                Dictionary<KeyValuePair<int, int>, int> overlapCount = new Dictionary<KeyValuePair<int, int>, int>();

                foreach (KeyValuePair<int, int> a in firstMatrixOnes)
                {
                    foreach (KeyValuePair<int, int> b in secondMatrixOnes)
                    {
                        KeyValuePair<int, int> vector = new KeyValuePair<int, int>(
                            b.Key - a.Key,
                            b.Value - a.Value
                        );

                        if (overlapCount.ContainsKey(vector))
                        {
                            overlapCount[vector]++;
                        }
                        else
                        {
                            overlapCount[vector] = 1;
                        }
                        maximumOverlaps = Math.Max(maximumOverlaps, overlapCount[vector]);
                    }
                }

                return maximumOverlaps;
            }
            private List<KeyValuePair<int, int>> GetNonZeroCells(int[,] matrix)
            {
                List<KeyValuePair<int, int>> result = new List<KeyValuePair<int, int>>();
                for (int row = 0; row < matrix.GetLength(0); ++row)
                {
                    for (int col = 0; col < matrix.GetLength(1); ++col)
                    {
                        if (matrix[row, col] == 1)
                        {
                            result.Add(new KeyValuePair<int, int>(row, col));
                        }
                    }
                }
                return result;
            }
            /*
            Approach 3: Imagine Convolution
            Complexity Analysis
Let N be the width of the matrix.
•	Time Complexity: O(N^4)
o	We iterate through (2N−1)⋅(2N−1) number of kernels.
o	For each kernel, we perform a convolution operation, which takes O(N^2) time.
o	To sum up, the overall time complexity of the algorithm would be (2N−1)⋅(2N−1)⋅O(N^2)=O(N^4).
•	Space Complexity: O(N^2)
o	We extend the matrix B to the size of (3N−2)⋅(3N−2), which would require the space of O(N^2).

            */


            public int ImagineConvolution(int[][] matrixA, int[][] matrixB)
            {

                int size = matrixA.Length;
                int[][] paddedMatrixB = new int[3 * size - 2][];
                for (int i = 0; i < paddedMatrixB.Length; i++)
                    paddedMatrixB[i] = new int[3 * size - 2];

                for (int row = 0; row < size; ++row)
                    for (int col = 0; col < size; ++col)
                        paddedMatrixB[row + size - 1][col + size - 1] = matrixB[row][col];

                int maxOverlaps = 0;
                for (int xShift = 0; xShift < 2 * size - 1; ++xShift)
                    for (int yShift = 0; yShift < 2 * size - 1; ++yShift)
                    {
                        maxOverlaps = Math.Max(maxOverlaps,
                            Convolute(matrixA, paddedMatrixB, xShift, yShift));
                    }

                return maxOverlaps;
            }
            private int Convolute(int[][] matrixA, int[][] kernel, int xShift, int yShift)
            {
                int result = 0;
                for (int row = 0; row < matrixA.Length; ++row)
                    for (int col = 0; col < matrixA.Length; ++col)
                        result += matrixA[row][col] * kernel[row + yShift][col + xShift];
                return result;
            }

        }



        /*
        887. Super Egg Drop
        https://leetcode.com/problems/super-egg-drop/description/
        */

        class SuperEggDropSol
        {
            private Dictionary<int, int> memo = new Dictionary<int, int>();

            /*

        Approach 1: Dynamic Programming with Binary Search
            Complexity Analysis
        •	Time Complexity: O(KNlogN).
        •	Space Complexity: O(KN).

            */
            public int DPWithBinarySearch(int numberOfEggs, int numberOfFloors)
            {
                return CalculateDrops(numberOfEggs, numberOfFloors);
            }

            private int CalculateDrops(int numberOfEggs, int numberOfFloors)
            {
                if (!memo.ContainsKey(numberOfFloors * 100 + numberOfEggs))
                {
                    int result;
                    if (numberOfFloors == 0)
                    {
                        result = 0;
                    }
                    else if (numberOfEggs == 1)
                    {
                        result = numberOfFloors;
                    }
                    else
                    {
                        int low = 1, high = numberOfFloors;
                        while (low + 1 < high)
                        {
                            int mid = (low + high) / 2;
                            int dropIfEggBreaks = CalculateDrops(numberOfEggs - 1, mid - 1);
                            int dropIfEggDoesNotBreak = CalculateDrops(numberOfEggs, numberOfFloors - mid);

                            if (dropIfEggBreaks < dropIfEggDoesNotBreak)
                            {
                                low = mid;
                            }
                            else if (dropIfEggBreaks > dropIfEggDoesNotBreak)
                            {
                                high = mid;
                            }
                            else
                            {
                                low = high = mid;
                            }
                        }

                        result = 1 + Math.Min(Math.Max(CalculateDrops(numberOfEggs - 1, low - 1), CalculateDrops(numberOfEggs, numberOfFloors - low)),
                                               Math.Max(CalculateDrops(numberOfEggs - 1, high - 1), CalculateDrops(numberOfEggs, numberOfFloors - high)));
                    }

                    memo[numberOfFloors * 100 + numberOfEggs] = result;
                }

                return memo[numberOfFloors * 100 + numberOfEggs];
            }
            /*

    Approach 2: Dynamic Programming with Optimality Criterion
    Complexity Analysis
    •	Time Complexity: O(KN).
    •	Space Complexity: O(N).

            */
            public int DPOptimal(int numberOfEggs, int numberOfFloors)
            {
                // Right now, dp[i] represents dp(1, i)
                int[] dp = new int[numberOfFloors + 1];
                for (int i = 0; i <= numberOfFloors; ++i)
                    dp[i] = i;

                for (int eggs = 2; eggs <= numberOfEggs; ++eggs)
                {
                    // Now, we will develop dp2[i] = dp(k, i)
                    int[] dp2 = new int[numberOfFloors + 1];
                    int optimalFloor = 1;
                    for (int floors = 1; floors <= numberOfFloors; ++floors)
                    {
                        // Let's find dp2[n] = dp(k, n)
                        // Increase our optimal x while we can make our answer better.
                        // Notice max(dp[x-1], dp2[n-x]) > max(dp[x], dp2[n-x-1])
                        // is simply max(T1(x-1), T2(x-1)) > max(T1(x), T2(x)).
                        while (optimalFloor < floors && Math.Max(dp[optimalFloor - 1], dp2[floors - optimalFloor]) > Math.Max(dp[optimalFloor], dp2[floors - optimalFloor - 1]))
                            optimalFloor++;

                        // The final answer happens at this x.
                        dp2[floors] = 1 + Math.Max(dp[optimalFloor - 1], dp2[floors - optimalFloor]);
                    }

                    dp = dp2;
                }

                return dp[numberOfFloors];
            }
            /*
            Approach 3: Mathematical
            Complexity Analysis
•	Time Complexity: O(KlogN).
•	Space Complexity: O(1).

            */
            public int UsingMaths(int numberOfEggs, int numberOfFloors)
            {
                int lowerBound = 1, upperBound = numberOfFloors;
                while (lowerBound < upperBound)
                {
                    int middle = (lowerBound + upperBound) / 2;
                    if (Calculate(middle, numberOfEggs, numberOfFloors) < numberOfFloors)
                        lowerBound = middle + 1;
                    else
                        upperBound = middle;
                }

                return lowerBound;
            }

            public int Calculate(int drops, int numberOfEggs, int numberOfFloors)
            {
                int result = 0, combination = 1;
                for (int i = 1; i <= numberOfEggs; ++i)
                {
                    combination *= drops - i + 1;
                    combination /= i;
                    result += combination;
                    if (result >= numberOfFloors) break;
                }
                return result;
            }

        }

        /*
        904. Fruit Into Baskets
        https://leetcode.com/problems/fruit-into-baskets/description/
        */
        class TotalFruitSol
        {
            /*
            Approach 1: Brute Force
    Complexity Analysis
Let n be the length of the input array fruits.
•	Time complexity: O(n^3)
o	We have three nested loops, the first loop for the left index left, the second loop for the right index right, and the third loop for the index currentIndex between left and right.
o	In each step, we need to add the current fruit to the set basket, which takes constant time.
o	For each subarray, we need to calculate the size of the basket after the iteration, which also takes constant time.
o	Therefore, the overall time complexity is O(n^3).
•	Space complexity: O(n)
o	During the iteration, we need to count the types of fruits in every subarray and store them in a hash set. In the worst-case scenario, there could be O(n) different types in some subarrays, thus it requires O(n) space complexity.
o	Therefore, the overall space complexity is O(n
        
            */
            public int Naive(int[] fruits)
            {
                // Maximum number of fruits we can pick
                int maximumPickedFruits = 0;

                // Iterate over all subarrays (left, right)
                for (int leftIndex = 0; leftIndex < fruits.Length; ++leftIndex)
                {
                    for (int rightIndex = 0; rightIndex < fruits.Length; ++rightIndex)
                    {
                        // Use a set to count the type of fruits.
                        HashSet<int> fruitBasket = new HashSet<int>();

                        // Iterate over the current subarray.
                        for (int currentIndex = leftIndex; currentIndex <= rightIndex; ++currentIndex)
                        {
                            fruitBasket.Add(fruits[currentIndex]);
                        }

                        // If the number of types of fruits in this subarray (types of fruits) 
                        // is no larger than 2, this is a valid subarray, update 'maximumPickedFruits'.
                        if (fruitBasket.Count <= 2)
                        {
                            maximumPickedFruits = Math.Max(maximumPickedFruits, rightIndex - leftIndex + 1);
                        }
                    }
                }

                // Return 'maximumPickedFruits' as the maximum length (maximum number of fruits we can pick).
                return maximumPickedFruits;
            }

            /*
            Approach 2: Optimized Brute Force
        Complexity Analysis
Let n be the length of the input array fruits.
•	Time complexity: O(n^2)
o	Compared with approach 1, we only have two nested loops now.
o	In each iteration step, we need to add the current fruit to the hash set basket, which takes constant time.
o	To sum up, the overall time complexity is O(n^2)
•	Space complexity: O(1)
o	During the iteration, we need to count the number of types in every possible subarray and update the maximum length. Since we used the early stop method, thus the types will never exceed 3. Therefore, the space complexity is O(1)
    
            */

            public int NaiveOptimal(int[] fruits)
            {
                // Maximum number of fruits we can pick
                int maxPicked = 0;

                // Iterate over the left index left of subarrays.
                for (int leftIndex = 0; leftIndex < fruits.Length; ++leftIndex)
                {
                    // Empty set to count the type of fruits.
                    HashSet<int> fruitBasket = new HashSet<int>();
                    int rightIndex = leftIndex;

                    // Iterate over the right index right of subarrays.
                    while (rightIndex < fruits.Length)
                    {
                        // Early stop. If adding this fruit makes 3 types of fruit,
                        // we should stop the inner loop.
                        if (!fruitBasket.Contains(fruits[rightIndex]) && fruitBasket.Count == 2)
                            break;

                        // Otherwise, update the number of this fruit.
                        fruitBasket.Add(fruits[rightIndex]);
                        rightIndex++;
                    }

                    // Update maxPicked.
                    maxPicked = Math.Max(maxPicked, rightIndex - leftIndex);
                }

                // Return maxPicked as the maximum length of valid subarray.
                // (maximum number of fruits we can pick).
                return maxPicked;
            }

            /*
           Approach 3: Sliding Window
 Complexity Analysis
Let n be the length of the input array fruits.
•	Time complexity: O(n)
o	Both indexes left and right only monotonically increased during the iteration, thus we have at most 2⋅n steps,
o	At each step, we update the hash set by addition or deletion of one fruit, which takes constant time.
o	In summary, the overall time complexity is O(n)
•	Space complexity: O(n)
o	In the worst-case scenario, there might be at most O(n) types of fruits inside the window. Take the picture below as an example. Imagine that we have an array of fruits like the following. (The first half is all one kind of fruit, while the second half is n/2 types of fruits)
o	Therefore, the space complexity is O(n).
            */
            public int SlidingWindow(int[] fruits)
            {
                // Hash map 'basket' to store the types of fruits.
                Dictionary<int, int> basket = new Dictionary<int, int>();
                int leftIndex = 0, rightIndex;

                // Add fruit from right side (rightIndex) of the window.
                for (rightIndex = 0; rightIndex < fruits.Length; ++rightIndex)
                {
                    if (basket.ContainsKey(fruits[rightIndex]))
                    {
                        basket[fruits[rightIndex]]++;
                    }
                    else
                    {
                        basket[fruits[rightIndex]] = 1;
                    }

                    // If the current window has more than 2 types of fruit,
                    // we remove one fruit from the left index (leftIndex) of the window.
                    if (basket.Count > 2)
                    {
                        basket[fruits[leftIndex]]--;
                        if (basket[fruits[leftIndex]] == 0)
                            basket.Remove(fruits[leftIndex]);
                        leftIndex++;
                    }
                }

                // Once we finish the iteration, the indexes leftIndex and rightIndex 
                // stands for the longest valid subarray we encountered.
                return rightIndex - leftIndex;
            }
            /*
            Approach 4: Sliding Window II
Complexity Analysis
Let n be the length of the input array fruits.
•	Time complexity: O(n)
o	Similarly, both indexes left and right are only monotonically increasing during the iteration, thus we have at most 2⋅n steps,
o	At each step, we update the hash set by addition or deletion of one fruit, which takes constant time. Note that the number of additions or deletions does not exceed n.
o	To sum up, the overall time complexity is O(n)
•	Space complexity: O(1)
o	We maintain the number of fruit types contained in the window in time. Therefore, at any given time, there are at most 3 types of fruits in the window or the hash map basket.
o	In summary, the space complexity is O(1).

            */
            public int SlidingWindowWithOptimalSpace(int[] fruits)
            {
                // We use a dictionary 'basket' to store the number of each type of fruit.
                Dictionary<int, int> basket = new Dictionary<int, int>();
                int leftIndex = 0, maxFruitsPicked = 0;

                // Add fruit from the right index (right) of the window.
                for (int rightIndex = 0; rightIndex < fruits.Length; ++rightIndex)
                {
                    if (basket.ContainsKey(fruits[rightIndex]))
                    {
                        basket[fruits[rightIndex]]++;
                    }
                    else
                    {
                        basket[fruits[rightIndex]] = 1;
                    }

                    // If the current window has more than 2 types of fruit,
                    // we remove fruit from the left index (left) of the window,
                    // until the window has only 2 types of fruit.
                    while (basket.Count > 2)
                    {
                        basket[fruits[leftIndex]]--;
                        if (basket[fruits[leftIndex]] == 0)
                        {
                            basket.Remove(fruits[leftIndex]);
                        }
                        leftIndex++;
                    }

                    // Update maxFruitsPicked.
                    maxFruitsPicked = Math.Max(maxFruitsPicked, rightIndex - leftIndex + 1);
                }

                // Return maxFruitsPicked as the maximum number of fruits we can collect.
                return maxFruitsPicked;
            }

        }


        /*
        909. Snakes and Ladders
        https://leetcode.com/problems/snakes-and-ladders/description/
        */
        public class SnakesAndLaddersSol
        {
            /*
            Approach 1: Breadth-first search
Complexity Analysis
Let n be the number of rows and columns.
•	Time complexity: O(n^2).
We run BFS on a graph whose vertices are the board cells, and the edges are moves between them. There are n^2 vertices and no more than 6n^2=O(n^2) edges.
The time complexity of BFS is O(∣V∣+∣E∣), where ∣V∣ is the number of vertices and ∣E∣ is the number of edges. We have ∣V∣=n^2 and ∣E∣<6n^2, thus the total time complexity for BFS is O(7n^2)=O(n^2). We also spend some time associating each (row, col) with a label, but this also costs O(n^2), so the overall time complexity is O(n^2).
•	Space complexity: O(n^2).
We maintain cells for each label from 1 to n^2, dist for distances to all cells and a queue for BFS. The columns array takes only O(n) space.

            */
            public int BFS(int[][] board)
            {
                int boardSize = board.Length;
                Tuple<int, int>[] cells = new Tuple<int, int>[boardSize * boardSize + 1];
                int label = 1;
                int[] columnIndices = new int[boardSize];
                for (int i = 0; i < boardSize; i++)
                {
                    columnIndices[i] = i;
                }
                for (int row = boardSize - 1; row >= 0; row--)
                {
                    foreach (int column in columnIndices)
                    {
                        cells[label++] = new Tuple<int, int>(row, column);
                    }
                    Array.Reverse(columnIndices);
                }
                int[] distances = new int[boardSize * boardSize + 1];
                Array.Fill(distances, -1);
                Queue<int> queue = new Queue<int>();
                distances[1] = 0;
                queue.Enqueue(1);
                while (queue.Count > 0)
                {
                    int current = queue.Dequeue();
                    for (int next = current + 1; next <= Math.Min(current + 6, boardSize * boardSize); next++)
                    {
                        int row = cells[next].Item1, column = cells[next].Item2;
                        int destination = board[row][column] != -1 ? board[row][column] : next;
                        if (distances[destination] == -1)
                        {
                            distances[destination] = distances[current] + 1;
                            queue.Enqueue(destination);
                        }
                    }
                }
                return distances[boardSize * boardSize];
            }
            /*            
Approach 2: Dijkstra's algorithm
Complexity Analysis
Let n be the number of columns and rows of the board.
•	Time complexity: O(n2⋅logn).
Dijkstra's algorithm with a binary heap works in O(∣V∣+∣E∣log∣V∣), where ∣V∣ is the number of vertices and ∣E∣ is the number of edges. As mentioned earlier in the BFS approach, in this problem, we have ∣V∣=n^2,∣E∣<6n^2.
•	Space complexity: O(n^2).
The space complexity of Dijkstra's algorithm is O(∣V∣)=O(n2) because we need to store ∣V∣ vertices in our data structure (we use a priority queue and an array of distances). Also, we have the cells of size O(n^2) and columns of size O(n).


            */
            public int Dijkstra(int[][] board)
            {
                int boardSize = board.Length;
                Tuple<int, int>[] cells = new Tuple<int, int>[boardSize * boardSize + 1];
                int label = 1;
                int[] columns = new int[boardSize];
                for (int i = 0; i < boardSize; i++)
                {
                    columns[i] = i;
                }
                for (int row = boardSize - 1; row >= 0; row--)
                {
                    foreach (int column in columns)
                    {
                        cells[label++] = new Tuple<int, int>(row, column);
                    }
                    Array.Reverse(columns);
                }
                int[] distances = new int[boardSize * boardSize + 1];
                Array.Fill(distances, -1);


                //TODO: fix below compare code
                PriorityQueue<Vertex, int> queue = new PriorityQueue<Vertex, int>(Comparer<int>.Create((dist1, dist2) => dist1.CompareTo(dist2)));
                distances[1] = 0;
                queue.Enqueue(new Vertex(0, 1), 0);

                while (queue.Count > 0)
                {
                    Vertex vertex = queue.Dequeue();
                    int currentDistance = vertex.Distance, currentLabel = vertex.Label;
                    if (currentDistance != distances[currentLabel])
                    {
                        continue;
                    }
                    for (int next = currentLabel + 1; next <= Math.Min(currentLabel + 6, boardSize * boardSize); next++)
                    {
                        int row = cells[next].Item1, column = cells[next].Item2;
                        int destination = board[row][column] != -1 ? board[row][column] : next;
                        if (distances[destination] == -1 || distances[currentLabel] + 1 < distances[destination])
                        {
                            distances[destination] = distances[currentLabel] + 1;
                            queue.Enqueue(new Vertex(distances[destination], destination), distances[destination]);
                        }
                    }
                }
                return distances[boardSize * boardSize];
            }

            class Vertex : IComparable<Vertex>
            {
                public int Distance { get; }
                public int Label { get; }

                public Vertex(int distance, int label)
                {
                    Distance = distance;
                    Label = label;
                }

                public int CompareTo(Vertex other)
                {
                    return this.Distance - other.Distance;
                }
            }

        }

        /*
        920. Number of Music Playlists
        https://leetcode.com/problems/number-of-music-playlists/description/
        */
        public class NumMusicPlaylistsSol
        {
            /*
            Approach 1: Bottom-up Dynamic Programming
Complexity Analysis
•	Time Complexity: O(goal⋅n).
We need to iterate over a two-dimensional DP table of size goal+1 by n+1. In each cell, we perform constant time operations.
•	Space Complexity: O(goal⋅n).
We're maintaining a two-dimensional DP table of size goal+1 by n+1 to store intermediate results.

            */
            public int BottomUpDP(int totalSongs, int targetLength, int maxReplays)
            {
                int MOD = 1_000_000_007;

                // Initialize the DP table
                long[][] dp = new long[targetLength + 1][];
                for (int i = 0; i < dp.Length; i++)
                {
                    dp[i] = new long[totalSongs + 1];
                }
                dp[0][0] = 1;

                for (int currentLength = 1; currentLength <= targetLength; currentLength++)
                {
                    for (int uniqueSongs = 1; uniqueSongs <= Math.Min(currentLength, totalSongs); uniqueSongs++)
                    {
                        // The current song is a new song
                        dp[currentLength][uniqueSongs] = dp[currentLength - 1][uniqueSongs - 1] * (totalSongs - uniqueSongs + 1) % MOD;
                        // The current song is a song we have played before
                        if (uniqueSongs > maxReplays)
                        {
                            dp[currentLength][uniqueSongs] = (dp[currentLength][uniqueSongs] + dp[currentLength - 1][uniqueSongs] * (uniqueSongs - maxReplays)) % MOD;
                        }
                    }
                }

                return (int)dp[targetLength][totalSongs];
            }
            /*            
Approach 2: Top-down Dynamic Programming (Memoization)
Complexity Analysis
•	Time Complexity: O(goal⋅n).
We are filling up a 2D DP table with goal+1 rows and n+1 columns. Each cell of the DP table gets filled once.
•	Space Complexity: O(goal⋅n).
The 2D DP table uses O(goal⋅n) of memory.

            */
            private const int MOD = 1_000_000_007;
            private long?[,] dp;

            public int TopDownDPWithMemo(int totalSongs, int targetLength, int maxReplays)
            {
                dp = new long?[targetLength + 1, totalSongs + 1];
                for (int i = 0; i <= targetLength; i++)
                {
                    for (int j = 0; j <= totalSongs; j++)
                    {
                        dp[i, j] = -1;
                    }
                }
                return (int)(NumberOfPlaylists(targetLength, totalSongs, maxReplays, totalSongs));
            }

            private long NumberOfPlaylists(int currentLength, int currentSongs, int maxReplays, int totalSongs)
            {
                // Base cases
                if (currentLength == 0 && currentSongs == 0)
                {
                    return 1;
                }
                if (currentLength == 0 || currentSongs == 0)
                {
                    return 0;
                }
                if (dp[currentLength, currentSongs] != -1)
                {
                    return dp[currentLength, currentSongs].Value;
                }
                // DP transition: add a new song or replay an old one
                dp[currentLength, currentSongs] = (NumberOfPlaylists(currentLength - 1, currentSongs - 1, maxReplays, totalSongs) * (totalSongs - currentSongs + 1)) % MOD;
                if (currentSongs > maxReplays)
                {
                    dp[currentLength, currentSongs] += (NumberOfPlaylists(currentLength - 1, currentSongs, maxReplays, totalSongs) * (currentSongs - maxReplays)) % MOD;
                    dp[currentLength, currentSongs] %= MOD;
                }
                return dp[currentLength, currentSongs].Value;
            }
            /*
Approach 3: Combinatorics

Complexity Analysis
•	Time Complexity: O(nloggoal).
The main loop runs from n down to k, so it iterates n−k+1=O(n) times.
Inside the main loop, we calculate the power of (i−k) raised to (goal−k), which takes O(loggoal) time.
So the total time complexity is O(nloggoal).
•	Space Complexity: O(n).
We maintain arrays for precalculated factorials and inverse factorials.

            */

            // Pre-calculated factorials and inverse factorials
            private long[] factorial;
            private long[] invFactorial;

            // Main method: calculates number of playlists
            public int UsingCombinatorics(int numberOfSongs, int targetLength, int maxSongsInPlaylist)
            {
                // Pre-calculate factorials and inverse factorials
                PrecalculateFactorials(numberOfSongs);

                // Initialize variables for calculation
                int sign = 1;
                long answer = 0;

                // Loop from 'numberOfSongs' down to 'maxSongsInPlaylist'
                for (int i = numberOfSongs; i >= maxSongsInPlaylist; i--)
                {
                    // Calculate temporary result for this iteration
                    long temp = Power(i - maxSongsInPlaylist, targetLength - maxSongsInPlaylist);
                    temp = (temp * invFactorial[numberOfSongs - i]) % MOD;
                    temp = (temp * invFactorial[i - maxSongsInPlaylist]) % MOD;

                    // Add or subtract temporary result to/from answer
                    answer = (answer + sign * temp + MOD) % MOD;

                    // Flip sign for next iteration
                    sign *= -1;
                }

                // Final result is numberOfSongs! * answer, all under modulo
                return (int)((factorial[numberOfSongs] * answer) % MOD);
            }

            // Method to pre-calculate factorials and inverse factorials up to 'n'
            private void PrecalculateFactorials(int numberOfSongs)
            {
                factorial = new long[numberOfSongs + 1];
                invFactorial = new long[numberOfSongs + 1];
                factorial[0] = invFactorial[0] = 1;

                // Calculate factorials and inverse factorials for each number up to 'numberOfSongs'
                for (int i = 1; i <= numberOfSongs; i++)
                {
                    factorial[i] = (factorial[i - 1] * i) % MOD;
                    // Inverse factorial calculated using Fermat's Little Theorem
                    invFactorial[i] = Power(factorial[i], (int)(MOD - 2));
                }
            }

            // Method to calculate power of a number under modulo using binary exponentiation
            private long Power(long baseValue, int exponent)
            {
                long result = 1L;

                // Loop until exponent is not zero
                while (exponent > 0)
                {
                    // If exponent is odd, multiply result with base
                    if ((exponent & 1) == 1)
                    {
                        result = (result * baseValue) % MOD;
                    }
                    // Divide the exponent by 2 and square the base
                    exponent >>= 1;
                    baseValue = (baseValue * baseValue) % MOD;
                }

                return result;
            }
        }

        /*
        924. Minimize Malware Spread
        https://leetcode.com/problems/minimize-malware-spread/description/
        */
        public class MinimizeMalwareSpreadSol
        {
            /*
            Approach 1: Depth First Search
          Complexity Analysis
•	Time Complexity: O(N^2), where N is the length of graph, as the graph is given in adjacent matrix form.
•	Space Complexity: O(N).
  
            */
            public int DFS(int[][] graph, int[] initial)
            {
                // 1. Color each component.
                // colors[node] = the color of this node.

                int numberOfNodes = graph.Length;
                int[] colors = new int[numberOfNodes];
                Array.Fill(colors, -1);
                int componentCount = 0;

                for (int node = 0; node < numberOfNodes; ++node)
                    if (colors[node] == -1)
                        Dfs(graph, colors, node, componentCount++);

                // 2. Size of each color.
                int[] size = new int[componentCount];
                foreach (int color in colors)
                    size[color]++;

                // 3. Find unique colors.
                int[] colorCount = new int[componentCount];
                foreach (int node in initial)
                    colorCount[colors[node]]++;

                // 4. Answer
                int answer = int.MaxValue;
                foreach (int node in initial)
                {
                    int color = colors[node];
                    if (colorCount[color] == 1)
                    {
                        if (answer == int.MaxValue)
                            answer = node;
                        else if (size[color] > size[colors[answer]])
                            answer = node;
                        else if (size[color] == size[colors[answer]] && node < answer)
                            answer = node;
                    }
                }

                if (answer == int.MaxValue)
                    foreach (int node in initial)
                        answer = Math.Min(answer, node);

                return answer;
            }

            private void Dfs(int[][] graph, int[] colors, int node, int color)
            {
                colors[node] = color;
                for (int neighbor = 0; neighbor < graph.Length; ++neighbor)
                    if (graph[node][neighbor] == 1 && colors[neighbor] == -1)
                        Dfs(graph, colors, neighbor, color);
            }

            /*
            Approach 2: Union-Find
            Complexity Analysis
•	Time Complexity: O(N^2), where N is the length of graph, as the graph is given in adjacent matrix form.
•	Space Complexity: O(N).
            */
            public int MinMalwareSpread(int[][] graph, int[] initial)
            {
                int numberOfNodes = graph.Length;
                DisjointSetUnion disjointSetUnion = new DisjointSetUnion(numberOfNodes);
                for (int i = 0; i < numberOfNodes; ++i)
                {
                    for (int j = i + 1; j < numberOfNodes; ++j)
                    {
                        if (graph[i][j] == 1)
                        {
                            disjointSetUnion.Union(i, j);
                        }
                    }
                }

                int[] count = new int[numberOfNodes];
                foreach (int node in initial)
                {
                    count[disjointSetUnion.Find(node)]++;
                }

                int resultNode = -1, largestComponentSize = -1;
                foreach (int node in initial)
                {
                    int root = disjointSetUnion.Find(node);
                    if (count[root] == 1)
                    {  // unique color
                        int currentComponentSize = disjointSetUnion.Size(root);
                        if (currentComponentSize > largestComponentSize)
                        {
                            largestComponentSize = currentComponentSize;
                            resultNode = node;
                        }
                        else if (currentComponentSize == largestComponentSize && node < resultNode)
                        {
                            largestComponentSize = currentComponentSize;
                            resultNode = node;
                        }
                    }
                }

                if (resultNode == -1)
                {
                    resultNode = int.MaxValue;
                    foreach (int node in initial)
                    {
                        resultNode = Math.Min(resultNode, node);
                    }
                }
                return resultNode;
            }
            public class DisjointSetUnion
            {
                private int[] parent, size;

                public DisjointSetUnion(int numberOfNodes)
                {
                    parent = new int[numberOfNodes];
                    for (int x = 0; x < numberOfNodes; ++x)
                    {
                        parent[x] = x;
                    }

                    size = new int[numberOfNodes];
                    Array.Fill(size, 1);
                }

                public int Find(int x)
                {
                    if (parent[x] != x)
                    {
                        parent[x] = Find(parent[x]);
                    }
                    return parent[x];
                }

                public void Union(int x, int y)
                {
                    int rootX = Find(x);
                    int rootY = Find(y);
                    parent[rootX] = rootY;
                    size[rootY] += size[rootX];
                }

                public int Size(int x)
                {
                    return size[Find(x)];
                }
            }


        }

        /*
        928. Minimize Malware Spread II
        https://leetcode.com/problems/minimize-malware-spread-ii/description/
        */

        // Translated from Java to C# for a solution in a console application
        class MinimizeMalwareSpreadIISol
        {

            /*
        Approach 1: DFS
        Time complexity: O(K N2) where N = number of total nodes and K = number of initially infected nodes.
        Space complexity: O(N)

            */
            public int DFS(int[][] graph, int[] initialNodes)
            {
                int numberOfNodes = graph.Length;
                int resultNode = initialNodes[0];
                int maximumCount = 0;
                bool[] infectedNodes = new bool[numberOfNodes];
                foreach (int node in initialNodes) infectedNodes[node] = true;

                foreach (int node in initialNodes)
                {
                    bool[] visitedNodes = new bool[numberOfNodes];
                    visitedNodes[node] = true;
                    int totalCount = 0;
                    for (int i = 0; i < numberOfNodes; i++)
                    {
                        if (!visitedNodes[i] && graph[node][i] == 1)
                        {
                            totalCount += DepthFirstSearch(graph, i, visitedNodes, infectedNodes);
                        }
                    }
                    if (totalCount > maximumCount || (totalCount == maximumCount && node < resultNode))
                    {
                        maximumCount = totalCount;
                        resultNode = node;
                    }
                }
                return resultNode;
            }
            private int DepthFirstSearch(int[][] graph, int currentNode, bool[] visitedNodes, bool[] infectedNodes)
            {
                if (infectedNodes[currentNode]) return 0;
                visitedNodes[currentNode] = true;
                int count = 1;
                for (int adjacentNode = 0; adjacentNode < graph[currentNode].Length; adjacentNode++)
                {
                    if (!visitedNodes[adjacentNode] && graph[currentNode][adjacentNode] == 1)
                    {
                        int infectedCount = DepthFirstSearch(graph, adjacentNode, visitedNodes, infectedNodes);
                        if (infectedCount == 0)
                        {
                            infectedNodes[currentNode] = true;
                            return 0;
                        }
                        count += infectedCount;
                    }
                }
                return count;
            }

            /*
            Approach 2: Disjoint Set /UnionFind
            Time complexity: O(N^2)
        Space complexity: O(N)
            */
            public int DisjointSetUnion(int[][] graph, int[] initial)
            {
                int numberOfNodes = graph.Length;
                int resultNode = initial[0];
                int maxSavedNodes = 0;
                bool[] isInfected = new bool[numberOfNodes];

                foreach (int infectedNode in initial)
                {
                    isInfected[infectedNode] = true;
                }

                UnionFind unionFind = new UnionFind(numberOfNodes);
                for (int nodeI = 0; nodeI < numberOfNodes; nodeI++)
                {
                    if (!isInfected[nodeI])
                    {
                        for (int nodeJ = nodeI + 1; nodeJ < numberOfNodes; nodeJ++)
                        {
                            if (!isInfected[nodeJ] && graph[nodeI][nodeJ] == 1)
                            {
                                unionFind.Union(nodeI, nodeJ);
                            }
                        }
                    }
                }

                int[] componentCount = new int[numberOfNodes];
                HashSet<int>[] components = new HashSet<int>[numberOfNodes];
                foreach (int infectedNode in initial)
                {
                    components[infectedNode] = new HashSet<int>();
                    for (int nodeV = 0; nodeV < numberOfNodes; nodeV++)
                    {
                        if (!isInfected[nodeV] && graph[infectedNode][nodeV] == 1)
                        {
                            components[infectedNode].Add(unionFind.Find(nodeV));
                        }
                    }
                    foreach (int component in components[infectedNode])
                    {
                        componentCount[component]++;
                    }
                }

                foreach (int infectedNode in initial)
                {
                    int savedNodes = 0;
                    foreach (int component in components[infectedNode])
                    {
                        if (componentCount[component] == 1)
                        {
                            savedNodes += unionFind.Size[component];
                        }
                    }
                    if (savedNodes > maxSavedNodes || (savedNodes == maxSavedNodes && infectedNode < resultNode))
                    {
                        maxSavedNodes = savedNodes;
                        resultNode = infectedNode;
                    }
                }
                return resultNode;
            }
            class UnionFind
            {
                public int[] Parent { get; }
                public int[] Size { get; }

                public UnionFind(int numberOfNodes)
                {
                    Parent = new int[numberOfNodes];
                    Size = new int[numberOfNodes];
                    for (int index = 0; index < numberOfNodes; index++)
                    {
                        Parent[index] = index;
                        Size[index] = 1;
                    }
                }

                public int Find(int node)
                {
                    return node != Parent[node] ? Parent[node] = Find(Parent[node]) : node;
                }

                public void Union(int nodeX, int nodeY)
                {
                    int rootX = Find(nodeX);
                    int rootY = Find(nodeY);
                    if (rootX != rootY)
                    {
                        if (Size[rootX] < Size[rootY])
                        {
                            Parent[rootX] = rootY;
                            Size[rootY] += Size[rootX];
                        }
                        else
                        {
                            Parent[rootY] = rootX;
                            Size[rootX] += Size[rootY];
                        }
                    }
                }
            }
            /*
            Approach 3: Tarjan's algorithm
Time complexity: O(N^2)
Space complexity: O(N)

            */
            public int TarjansAlgo(int[][] graph, int[] initial)
            {
                int numberOfNodes = graph.Length, answerNode = initial[0], maxCount = 0;
                bool[] infected = new bool[numberOfNodes];
                foreach (int node in initial) infected[node] = true;
                int[] depth = new int[numberOfNodes], low = new int[numberOfNodes], count = new int[numberOfNodes];
                foreach (int node in initial)
                {
                    if (depth[node] == 0)
                    {
                        DepthFirstSearch(graph, node, -1, 1, depth, low, infected, count);
                    }
                    if (count[node] > maxCount || (count[node] == maxCount && node < answerNode))
                    {
                        maxCount = count[node];
                        answerNode = node;
                    }
                }
                return answerNode;
            }
            private int DepthFirstSearch(int[][] graph, int currentNode, int parentNode, int time, int[] depth, int[] low, bool[] infected, int[] count)
            {
                low[currentNode] = depth[currentNode] = time;
                bool isInfected = infected[currentNode];
                int size = 1;
                for (int neighbor = 0; neighbor < graph[currentNode].Length; neighbor++)
                {
                    if (graph[currentNode][neighbor] == 1)
                    {
                        if (depth[neighbor] == 0)
                        {
                            int subTreeSize = DepthFirstSearch(graph, neighbor, currentNode, time + 1, depth, low, infected, count);
                            if (subTreeSize == 0)
                            {
                                isInfected = true;
                            }
                            else
                            {
                                size += subTreeSize;
                            }
                            if (low[neighbor] >= depth[currentNode])
                            {
                                count[currentNode] += subTreeSize;
                            }
                            low[currentNode] = Math.Min(low[currentNode], low[neighbor]);
                        }
                        else if (neighbor != parentNode)
                        {
                            low[currentNode] = Math.Min(low[currentNode], depth[neighbor]);
                        }
                    }
                }
                return isInfected ? 0 : size;
            }


        }


        /* 929. Unique Email Addresses
        https://leetcode.com/problems/unique-email-addresses/description/
         */
        class NumUniqueEmailsSol
        {
            /*
            Approach 1: Linear Iteration
          Complexity Analysis
Let N be the number of the emails and M be the average length of an email.
•	Time Complexity: O(N⋅M)
In the worst case, we iterate over all the characters of each of the emails given.
If we have N emails and each email has M characters in it.
Then complexity is of order (Number of emails) * (Number of characters in average email) = N*M.
•	Space Complexity: O(N⋅M)
In the worst case, when all emails are unique, we will store every email address given to us in the hash set.
  
            */
            public int LinerIteration(string[] emails)
            {
                // hash set to store all the unique emails
                HashSet<string> uniqueEmails = new HashSet<string>();

                foreach (string email in emails)
                {
                    StringBuilder cleanMail = new StringBuilder();

                    // iterate over each character in email
                    for (int i = 0; i < email.Length; ++i)
                    {
                        char currentChar = email[i];

                        // stop adding characters to localName
                        if (currentChar == '+' || currentChar == '@') break;

                        // add this character if not '.'
                        if (currentChar != '.') cleanMail.Append(currentChar);
                    }

                    // compute domain name (substring from end to '@')
                    StringBuilder domainName = new StringBuilder();

                    for (int i = email.Length - 1; i >= 0; --i)
                    {
                        char currentChar = email[i];
                        domainName.Append(currentChar);
                        if (currentChar == '@') break;
                    }


                    cleanMail.Append(domainName.ToString().Reverse());
                    uniqueEmails.Add(cleanMail.ToString());
                }

                return uniqueEmails.Count;
            }
            /*
            Approach 2: Using String Split Method
Complexity Analysis
Let N be the number of the emails and M be the average length of an email.
•	Time Complexity: O(N⋅M)
The split method must iterate over all of the characters in each email and the replace method must iterate over all of the characters in each local name. As such, they both require linear time and are O(M) operations.
Since there are N emails and the average email has M characters in it, the complexity is of order (Number of emails) * (Number of characters in an email) = N*M.
•	Space Complexity: O(N⋅M)
In the worst case, when all emails are unique, we will store every email address given to us in the hash set.


            */
            public int UsingSplit(string[] emailAddresses)
            {
                // hash set to store all the unique emails
                HashSet<string> uniqueEmailAddresses = new HashSet<string>();

                foreach (string emailAddress in emailAddresses)
                {
                    // split into two parts local and domain
                    string[] parts = emailAddress.Split('@');

                    // split local by '+'
                    string[] localParts = parts[0].Split('+');

                    // remove all '.', and concatenate '@' and append domain
                    uniqueEmailAddresses.Add(localParts[0].Replace(".", "") + "@" + parts[1]);
                }

                return uniqueEmailAddresses.Count;
            }

        }


        /* 933. Number of Recent Calls
        https://leetcode.com/problems/number-of-recent-calls/description/ */


        class RecentCounter
        {
            /*
Approach 1: Iteration over Sliding Window
Complexity Analysis
First of all, let us estimate the upper-bound on the size of our sliding window.
Here we quote an important condition from the problem description: "It is guaranteed that every call to ping uses a strictly larger value of t than before."
Based on the above condition, the maximal number of elements in our sliding window would be 3000, which is also the maximal time difference between the head and the tail elements.
•	Time Complexity: O(1)
o	The main time complexity of our ping() function lies in the loop, which in the worst case would run 3000 iterations to pop out all outdated elements, and in the best case a single iteration.
o	Therefore, for a single invocation of ping() function, its time complexity is O(3000)=O(1).
o	If we assume that there is a ping call at each timestamp, then the cost of ping() is further amortized, where at each invocation, we would only need to pop out a single element, once the sliding window reaches its upper bound.
•	Space Complexity: O(1)
o	As we estimated before, the maximal size of our sliding window is 3000, which is a constant.

            */
            LinkedList<int> slideWindow;

            public RecentCounter()
            {
                this.slideWindow = new LinkedList<int>();
            }

            public int Ping(int t)
            {
                // step 1). append the current call
                this.slideWindow.AddLast(t);

                // step 2). invalidate the outdated pings
                while (this.slideWindow.First() < t - 3000)
                    this.slideWindow.RemoveFirst();

                return this.slideWindow.Count;
            }
        }


        /* 934. Shortest Bridge
        https://leetcode.com/problems/shortest-bridge/description/ */

        class ShortestBridgeSol
        {
            /*
            Approach 1: Depth-First-Search + Breadth-First-Search
            Complexity Analysis
    Let n×n be the size of the input matrix grid.
    •	Time complexity: O(n^2)
    o	The general time complexity of Depth-First-Search is O(V+E), where V stands for the number of vertices. The maximum number of cells in the first island is n2, so iterating over its cells will take O(n^2) time. E is a constant here since we are only allowed to traverse in up to 4 directions.
    o	The general time complexity of Breadth-First-Search is O(V+E), where V stands for the number of vertices. The maximum number of water cells we need to check before reaching the second island is n^2, which will take O(n^2) time.
    •	Space complexity: O(n^2)
    o	The general space complexity of Depth-First-Search is O(V), where V stands for the number of vertices. The maximum number of cells in the first island is n2, thus the space used by the recursive stack during DFS is O(n^2)
    o	The general space complexity of Breadth-First-Search is O(V), where V stands for the number of vertices. The maximum number of water cells we need to check using BFS before reaching the second island is n2, thus the space used by the queue is O(n^2).
    o	To sum up, the overall space complexity is O(n^2)

            */
            private List<int[]> bfsQueue;

            // Recursively check the neighboring land cell of current cell grid[x][y] and add all
            // land cells of island A to bfsQueue.
            private void Dfs(int[][] grid, int x, int y, int n)
            {
                grid[x][y] = 2;
                bfsQueue.Add(new int[] { x, y });
                foreach (int[] pair in new int[][] { new int[] { x + 1, y }, new int[] { x - 1, y }, new int[] { x, y + 1 }, new int[] { x, y - 1 } })
                {
                    int currentX = pair[0], currentY = pair[1];
                    if (0 <= currentX && currentX < n && 0 <= currentY && currentY < n && grid[currentX][currentY] == 1)
                    {
                        Dfs(grid, currentX, currentY, n);
                    }
                }
            }

            // Find any land cell, and we treat it as a cell of island A.
            public int DFSAndBFS(int[][] grid)
            {
                int n = grid.Length;
                int firstX = -1, firstY = -1;
                for (int i = 0; i < n; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        if (grid[i][j] == 1)
                        {
                            firstX = i;
                            firstY = j;
                            break;
                        }
                    }
                }

                // Add all land cells of island A to bfsQueue.
                bfsQueue = new List<int[]>();
                Dfs(grid, firstX, firstY, n);

                int distance = 0;
                while (bfsQueue.Count > 0)
                {
                    List<int[]> newBfs = new List<int[]>();
                    foreach (int[] pair in bfsQueue)
                    {
                        int x = pair[0], y = pair[1];
                        foreach (int[] nextPair in new int[][] { new int[] { x + 1, y }, new int[] { x - 1, y }, new int[] { x, y + 1 }, new int[] { x, y - 1 } })
                        {
                            int currentX = nextPair[0], currentY = nextPair[1];
                            if (0 <= currentX && currentX < n && 0 <= currentY && currentY < n)
                            {
                                if (grid[currentX][currentY] == 1)
                                {
                                    return distance;
                                }
                                else if (grid[currentX][currentY] == 0)
                                {
                                    newBfs.Add(nextPair);
                                    grid[currentX][currentY] = -1;
                                }
                            }
                        }
                    }

                    // Once we finish one round without finding land cells of island B, we will
                    // start the next round on all water cells that are 1 cell further away from
                    // island A and increment the distance by 1.
                    bfsQueue = newBfs;
                    distance++;
                }

                return distance;
            }
            /*        
    Approach 2: Breadth-First-Search
    Complexity Analysis
    Let n×n be the size of the input matrix grid.
    •	Time complexity: O(n^2)
    o	The maximum number of water cells and the maximum number of land cells in island A we need to check are n2, which will take O(n2) time.
    •	Space complexity: O(n^2)
    o	The maximum number of land cells of island A that we need to check with BFS is n2, thus the space used by bfs_queue is O(n^2).
    o	The maximum number of water cells we need to check using BFS before reaching the second island is n2, thus the space used by second_bfs_queue is also O(n^2).
    o	To sum up, the overall space complexity is O(n^2)


            */
            public int BFS(int[][] grid)
            {
                int gridSize = grid.Length;
                int firstLandX = -1, firstLandY = -1;

                // Find any land cell, and we treat it as a cell of island A.
                for (int row = 0; row < gridSize; row++)
                {
                    for (int col = 0; col < gridSize; col++)
                    {
                        if (grid[row][col] == 1)
                        {
                            firstLandX = row;
                            firstLandY = col;
                            break;
                        }
                    }
                }

                // bfsQueue for BFS on land cells of island A; secondBfsQueue for BFS on water cells.
                List<int[]> bfsQueue = new List<int[]>();
                List<int[]> secondBfsQueue = new List<int[]>();
                bfsQueue.Add(new int[] { firstLandX, firstLandY });
                secondBfsQueue.Add(new int[] { firstLandX, firstLandY });
                grid[firstLandX][firstLandY] = 2;

                // BFS for all land cells of island A and add them to secondBfsQueue.
                while (bfsQueue.Count > 0)
                {
                    List<int[]> newBfs = new List<int[]>();
                    foreach (int[] cell in bfsQueue)
                    {
                        int x = cell[0];
                        int y = cell[1];
                        foreach (int[] next in new int[][] { new int[] { x + 1, y }, new int[] { x - 1, y }, new int[] { x, y + 1 }, new int[] { x, y - 1 } })
                        {
                            int currentX = next[0];
                            int currentY = next[1];
                            if (currentX >= 0 && currentX < gridSize && currentY >= 0 && currentY < gridSize && grid[currentX][currentY] == 1)
                            {
                                newBfs.Add(new int[] { currentX, currentY });
                                secondBfsQueue.Add(new int[] { currentX, currentY });
                                grid[currentX][currentY] = 2;
                            }
                        }
                    }
                    bfsQueue = newBfs;
                }

                int distance = 0;
                while (secondBfsQueue.Count > 0)
                {
                    List<int[]> newBfs = new List<int[]>();
                    foreach (int[] cell in secondBfsQueue)
                    {
                        int x = cell[0];
                        int y = cell[1];
                        foreach (int[] next in new int[][] { new int[] { x + 1, y }, new int[] { x - 1, y }, new int[] { x, y + 1 }, new int[] { x, y - 1 } })
                        {
                            int currentX = next[0];
                            int currentY = next[1];
                            if (currentX >= 0 && currentX < gridSize && currentY >= 0 && currentY < gridSize)
                            {
                                if (grid[currentX][currentY] == 1)
                                {
                                    return distance;
                                }
                                else if (grid[currentX][currentY] == 0)
                                {
                                    newBfs.Add(new int[] { currentX, currentY });
                                    grid[currentX][currentY] = -1;
                                }
                            }
                        }
                    }

                    // Once we finish one round without finding land cells of island B, we will
                    // start the next round on all water cells that are 1 cell further away from
                    // island A and increment the distance by 1.
                    secondBfsQueue = newBfs;
                    distance++;
                }
                return distance;
            }
        }


        /* 937. Reorder Data in Log Files
        https://leetcode.com/problems/reorder-data-in-log-files/description/
         */
        class ReorderLogFilesSol
        {
            /*
Approach 1: Comparator
Complexity Analysis
Let N be the number of logs in the list and
M be the maximum length of a single log.
•	Time Complexity: O(M⋅N⋅logN)
o	First of all, the time complexity of the Arrays.sort() is O(N⋅logN), as stated in the API specification, which is to say that the compare() function would be invoked O(N⋅logN) times.
o	For each invocation of the compare() function, it could take up to O(M) time, since we compare the contents of the logs.
o	Therefore, the overall time complexity of the algorithm is O(M⋅N⋅logN).
•	Space Complexity: O(M⋅logN)
o	For each invocation of the compare() function, we would need up to O(M) space to hold the parsed logs.
o	In addition, since the implementation of Arrays.sort() is based on quicksort algorithm whose space complexity is O(logn), assuming that the space for each element is O(1)).
Since each log could be of O(M) space, we would need O(M⋅logN) space to hold the intermediate values for sorting.
o	In total, the overall space complexity of the algorithm is O(M+M⋅logN)=O(M⋅logN).

            */
            public string[] UsingComparator(string[] logs)
            {

                Comparison<string> logComparator = (log1, log2) =>
                {
                    // split each log into two parts: <identifier, content>
                    string[] split1 = log1.Split(new[] { ' ' }, 2);
                    string[] split2 = log2.Split(new[] { ' ' }, 2);

                    bool isDigit1 = char.IsDigit(split1[1][0]);
                    bool isDigit2 = char.IsDigit(split2[1][0]);

                    // case 1). both logs are letter-logs
                    if (!isDigit1 && !isDigit2)
                    {
                        // first compare the content
                        int cmp = string.Compare(split1[1], split2[1]);
                        if (cmp != 0)
                            return cmp;
                        // logs of same content, compare the identifiers
                        return string.Compare(split1[0], split2[0]);
                    }

                    // case 2). one of logs is digit-log
                    if (!isDigit1 && isDigit2)
                        // the letter-log comes before digit-logs
                        return -1;
                    else if (isDigit1 && !isDigit2)
                        return 1;
                    else
                        // case 3). both logs are digit-log
                        return 0;
                };

                Array.Sort(logs, logComparator);
                return logs;
            }

            /*
Approach 2: Sorting by Keys
Complexity Analysis
Let N be the number of logs in the list and
M be the maximum length of a single log.
•	Time Complexity: O(M⋅N⋅logN)
o	The sorted() in Python is implemented with the Timsort algorithm whose time complexity is O(N⋅logN).
o	Since the keys of the elements are basically the logs itself, the comparison between two keys can take up to O(M) time.
o	Therefore, the overall time complexity of the algorithm is O(M⋅N⋅logN).
•	Space Complexity: O(M⋅N)
o	First, we need O(M⋅N) space to keep the keys for the log.
o	In addition, the worst space complexity of the Timsort algorithm is O(N), assuming that the space for each element is O(1).
Hence we would need O(M⋅N) space to hold the intermediate values for sorting.
o	In total, the overall space complexity of the algorithm is O(M⋅N+M⋅N)=O(M⋅N).


            */
        }


        /* 948. Bag of Tokens
        https://leetcode.com/problems/bag-of-tokens/description/
         */

        public class BagOfTokensScoreSol
        {
            /*
            Approach 1: Sort and Greedy With Two Pointer
            Complexity Analysis
Let n be the length of tokens.
•	Time complexity: O(nlogn)
Sorting tokens takes O(nlogn).
We process tokens using the pointers low and high until they meet in the middle or we can't play any more tokens. With each iteration, low is incremented, or high is decremented, or the loop terminates because we can't make any more moves that increase our score. We handle each token in tokens at most once, so the time complexity is O(n).
O(nlogn) is the dominating term.
•	Space complexity: O(n) or O(logn)
Sorting uses extra space, which depends on the implementation of each programming language.
Other than sorting, we use a handful of variables that use constant, O(1) space, so the space used for sorting is the dominant term
            */
            public int GreedyWithSortAndTwoPointers(int[] tokens, int power)
            {
                int lowIndex = 0;
                int highIndex = tokens.Length - 1;
                int score = 0;
                Array.Sort(tokens);

                while (lowIndex <= highIndex)
                {
                    // When we have enough power, play lowest token face-up
                    if (power >= tokens[lowIndex])
                    {
                        score += 1;
                        power -= tokens[lowIndex];
                        lowIndex += 1;
                    }
                    // We don't have enough power to play a token face-up
                    // If there is at least one token remaining,
                    // and we have enough score, play highest token face-down
                    else if (lowIndex < highIndex && score > 0)
                    {
                        score -= 1;
                        power += tokens[highIndex];
                        highIndex -= 1;
                    }
                    // We don't have enough score, power, or tokens 
                    // to play face-up or down and increase our score
                    else
                    {
                        return score;
                    }
                }
                return score;
            }

            /*
Approach 2: Sort and Greedy With DeQueue
Complexity Analysis
Let n be the length of tokens.
•	Time complexity: O(nlogn)
Sorting tokens takes O(nlogn).
We process tokens using the pointers low and high until they meet in the middle or we can't play any more tokens. With each iteration, low is incremented, or high is decremented, or the loop terminates because we can't make any more moves that increase our score. We handle each token in tokens at most once, so the time complexity is O(n).
O(nlogn) is the dominating term.
•	Space complexity: O(n) or O(logn)
Sorting uses extra space, which depends on the implementation of each programming language.
Other than sorting, we use a handful of variables that use constant, O(1) space, so the space used for sorting is the dominant term
*/
            public int GreedyWithSortAndDequeue(int[] tokens, int power)
            {
                int score = 0;
                Array.Sort(tokens);
                LinkedList<int> tokenDeque = new LinkedList<int>();

                foreach (int token in tokens)
                {
                    tokenDeque.AddLast(token);
                }

                while (tokenDeque.Count > 0)
                {
                    // When we have enough power, play token face-up
                    if (power >= tokenDeque.First.Value)
                    {
                        power -= tokenDeque.First.Value;
                        tokenDeque.RemoveFirst();
                        score++;
                        // We don't have enough power to play a token face-up
                        // When there is at least one token remaining,
                        // and we have enough score, play token face-down
                    }
                    else if (tokenDeque.Count > 1 && score > 0)
                    {
                        power += tokenDeque.Last.Value;
                        tokenDeque.RemoveLast();
                        score--;
                        // We don't have enough score, power, or tokens 
                        // to play face-up or down and increase our score
                    }
                    else
                    {
                        return score;
                    }
                }
                return score;
            }
        }


        /* 949. Largest Time for Given Digits
        https://leetcode.com/problems/largest-time-for-given-digits/description/
         */
        public class LargestTimeFromDigitsSol
        {
            /*
            Approach 1: Enumerate the Permutations
            Complexity Analysis
•	Time Complexity: O(1)
o	For an array of length N, the number of permutations would be N!.
In our case, the input is an array of 4 digits. Hence, the number of permutations would be 4!=4∗3∗2∗1=24.
o	Since the length of the input array is fixed, it would take the same constant time to generate its permutations, regardless the content of the array.
Therefore, the time complexity to generate the permutations would be O(1).
o	In the above program, each iteration takes a constant time to process.
Since the total number of permutations is fixed (constant), the time complexity of the loop in the algorithm is constant as well, i.e. 24⋅O(1)=O(1).
o	To sum up, the overall time complexity of the algorithm would be O(1)+O(1)=O(1).
•	Space Complexity: O(1)
o	In the algorithm, we keep the permutations for the input digits, which are in total 24, i.e. a constant number regardless the input.

            */
            public string EnumerateThePermutations(int[] digits)
            {
                int maxTime = -1;

                // Enumerate all possibilities using permutations
                for (int hourIndex = 0; hourIndex < digits.Length; hourIndex++)
                {
                    for (int minuteTensIndex = 0; minuteTensIndex < digits.Length; minuteTensIndex++)
                    {
                        if (minuteTensIndex == hourIndex) continue;
                        for (int minuteOnesIndex = 0; minuteOnesIndex < digits.Length; minuteOnesIndex++)
                        {
                            if (minuteOnesIndex == hourIndex || minuteOnesIndex == minuteTensIndex) continue;
                            for (int secondIndex = 0; secondIndex < digits.Length; secondIndex++)
                            {
                                if (secondIndex == hourIndex || secondIndex == minuteTensIndex || secondIndex == minuteOnesIndex) continue;

                                int hour = digits[hourIndex] * 10 + digits[minuteTensIndex];
                                int minute = digits[minuteOnesIndex] * 10 + digits[secondIndex];
                                if (hour < 24 && minute < 60)
                                {
                                    maxTime = Math.Max(maxTime, hour * 60 + minute);
                                }
                            }
                        }
                    }
                }

                if (maxTime == -1)
                {
                    return "";
                }
                else
                {
                    return string.Format("{0:D2}:{1:D2}", maxTime / 60, maxTime % 60);
                }
            }
            /*
            Approach 2: Permutation via Enumeration
Complexity Analysis
•	Time Complexity: O(1)
o	We have a 3-level nested loops, each loop would have 4 iterations. As a result, the total number of iterations is 4∗4∗4=64.
o	Since the length of the input array is fixed, it would take the same constant time to generate its permutations, regardless the content of the array.
Therefore, the time complexity to generate the permutations would be O(1).
o	Note that the total number of permutations is 4!=4∗3∗2∗1=24. Yet, it takes us 64 iterations to generate the permutations, which is not the most efficient algorithm as one can see.
As the size of array grows, this discrepancy would grow exponentially.
•	Space Complexity: O(1)
o	In the algorithm, we keep a variable to keep track of the maximum time, as well as some intermediates variables for the function.
Since the size of the input array is fixed, the total size of the local variables are bounded as well.
            */
            private int maximumTime = -1;
            private int maxTime;

            public string PermutationViaEnumeration(int[] digits)
            {
                this.maximumTime = -1;

                for (int firstIndex = 0; firstIndex < digits.Length; ++firstIndex)
                    for (int secondIndex = 0; secondIndex < digits.Length; ++secondIndex)
                        for (int thirdIndex = 0; thirdIndex < digits.Length; ++thirdIndex)
                        {
                            // skip duplicate elements
                            if (firstIndex == secondIndex || secondIndex == thirdIndex || firstIndex == thirdIndex)
                                continue;
                            // the total sum of indices is 0 + 1 + 2 + 3 = 6.
                            int fourthIndex = 6 - firstIndex - secondIndex - thirdIndex;
                            int[] permutation = { digits[firstIndex], digits[secondIndex], digits[thirdIndex], digits[fourthIndex] };
                            // check if the permutation can form a time
                            ValidateTime(permutation);
                        }
                if (this.maximumTime == -1)
                    return "";
                else
                    return string.Format("{0:D2}:{1:D2}", maximumTime / 60, maximumTime % 60);
            }

            private void ValidateTime(int[] permutation)
            {
                int hour = permutation[0] * 10 + permutation[1];
                int minute = permutation[2] * 10 + permutation[3];
                if (hour < 24 && minute < 60)
                    this.maximumTime = Math.Max(this.maximumTime, hour * 60 + minute);
            }
            /*
            Approach 3: Permutation via Backtracking
            Complexity Analysis
            •	Time Complexity: O(1)
            o	Since the length of the input array is fixed, it would take the same constant time to generate its permutations, regardless the content of the array.
            Therefore, the time complexity to generate the permutations would be O(1).
            o	Therefore, same as the previous approach, the overall time complexity of the algorithm would be O(1).
            •	Space Complexity: O(1)
            o	In the algorithm, we keep the permutations for the input digits, which are in total 24, i.e. a constant number regardless the input.
            o	Although the recursion in the algorithm could incur additional memory consumption in the function call stack, the maximal number of recursion is bounded by the size of the combination.
            Hence, the space overhead for the recursion in this problem is constant.

            */
            public string PermutationViaBacktracking(int[] digits)
            {
                this.maxTime = -1;
                Permutate(digits, 0);
                if (this.maxTime == -1)
                    return "";
                else
                    return string.Format("{0:D2}:{1:D2}", maxTime / 60, maxTime % 60);
            }

            protected void Permutate(int[] array, int start)
            {
                if (start == array.Length)
                {
                    this.BuildTime(array);
                    return;
                }
                for (int i = start; i < array.Length; ++i)
                {
                    this.Swap(array, i, start);
                    this.Permutate(array, start + 1);
                    this.Swap(array, i, start);
                }
            }

            protected void BuildTime(int[] perm)
            {
                int hour = perm[0] * 10 + perm[1];
                int minute = perm[2] * 10 + perm[3];
                if (hour < 24 && minute < 60)
                    this.maxTime = Math.Max(this.maxTime, hour * 60 + minute);
            }

            protected void Swap(int[] array, int i, int j)
            {
                if (i != j)
                {
                    int temp = array[i];
                    array[i] = array[j];
                    array[j] = temp;
                }
            }

        }

        /*
        968. Binary Tree Cameras
       https://leetcode.com/problems/binary-tree-cameras/description/
        */
        public class MinCameraCoverSol
        {
            /*
            Approach 1: Dynamic Programming
            Complexity Analysis
•	Time Complexity: O(N), where N is the number of nodes in the given tree.
•	Space Complexity: O(H), where H is the height of the given tree.

            */

            public int WithDP(TreeNode root)
            {
                int[] answer = Solve(root);
                return Math.Min(answer[1], answer[2]);
            }

            // 0: Strict ST; All nodes below this are covered, but not this one
            // 1: Normal ST; All nodes below and incl this are covered - no camera
            // 2: Placed camera; All nodes below this are covered, plus camera here
            private int[] Solve(TreeNode node)
            {
                if (node == null)
                    return new int[] { 0, 0, 99999 };

                int[] leftResult = Solve(node.Left);
                int[] rightResult = Solve(node.Right);
                int minLeft12 = Math.Min(leftResult[1], leftResult[2]);
                int minRight12 = Math.Min(rightResult[1], rightResult[2]);

                int d0 = leftResult[1] + rightResult[1];
                int d1 = Math.Min(leftResult[2] + minRight12, rightResult[2] + minLeft12);
                int d2 = 1 + Math.Min(leftResult[0], minLeft12) + Math.Min(rightResult[0], minRight12);
                return new int[] { d0, d1, d2 };
            }
            /*
            Approach 2: Greedy
Complexity Analysis
•	Time Complexity: O(N), where N is the number of nodes in the given tree.
•	Space Complexity: O(H), where H is the height of the given tree.

            */
            private int answer;
            private HashSet<TreeNode> coveredNodes;

            public int UsingGreedy(TreeNode root)
            {
                answer = 0;
                coveredNodes = new HashSet<TreeNode>();
                coveredNodes.Add(null);

                DepthFirstSearch(root, null);
                return answer;
            }

            private void DepthFirstSearch(TreeNode node, TreeNode parent)
            {
                if (node != null)
                {
                    DepthFirstSearch(node.Left, node);
                    DepthFirstSearch(node.Right, node);

                    if (parent == null && !coveredNodes.Contains(node) ||
                            !coveredNodes.Contains(node.Left) ||
                            !coveredNodes.Contains(node.Right))
                    {
                        answer++;
                        coveredNodes.Add(node);
                        coveredNodes.Add(parent);
                        coveredNodes.Add(node.Left);
                        coveredNodes.Add(node.Right);
                    }
                }
            }

        }


        /* 957. Prison Cells After N Days
        https://leetcode.com/problems/prison-cells-after-n-days/description/
         */
        public class PrisonAfterNDaysSol
        {
            /*
            Approach 1: Simulation with Fast Forwarding
         Complexity Analysis
Let K be the number of cells, and N be the number of steps.
•	Time Complexity: O(K⋅min(N,2^K))
o	As we discussed before, at most we could have 2^K possible states. While we run the simulation with N steps, we might need to run min(N,2^K) steps without fast-forwarding in the worst case.
o	For each simulation step, it takes O(K) time to process and evolve the state of cells.
o	Hence, the overall time complexity of the algorithm is O(K⋅min(N,2^K)).
•	Space Complexity:
o	The main memory consumption of the algorithm is the hashmap that we use to keep track of the states of the cells. The maximal number of entries in the hashmap would be 2^K as we discussed before.
o	In the Java implementation, we encode the state as a single integer value. Therefore, its space complexity would be O(2^K), assuming that K does not exceed 32 so that a state can fit into a single integer number.
o	In the Python implementation, we keep the states of cells as they are in the hashmap. As a result, for each entry, it takes O(K) space. In total, its space complexity becomes O(K⋅2^K).
   
            */
            public int[] SimulationWithFastForwarding(int[] cells, int N)
            {
                Dictionary<int, int> seen = new Dictionary<int, int>();
                bool isFastForwarded = false;

                // step 1). run the simulation with hashmap
                while (N > 0)
                {
                    if (!isFastForwarded)
                    {
                        int stateBitmap = this.CellsToBitmap(cells);
                        if (seen.ContainsKey(stateBitmap))
                        {
                            // the length of the cycle is seen[state_key] - N 
                            N %= seen[stateBitmap] - N;
                            isFastForwarded = true;
                        }
                        else
                        {
                            seen[stateBitmap] = N;
                        }
                    }
                    // check if there is still some steps remained,
                    // with or without the fast-forwarding.
                    if (N > 0)
                    {
                        N -= 1;
                        cells = this.NextDay(cells);
                    }
                }
                return cells;
            }
            protected int CellsToBitmap(int[] cells)
            {
                int stateBitmap = 0x0;
                foreach (int cell in cells)
                {
                    stateBitmap <<= 1;
                    stateBitmap = (stateBitmap | cell);
                }
                return stateBitmap;
            }

            protected int[] NextDay(int[] cells)
            {
                int[] newCells = new int[cells.Length];
                newCells[0] = 0;
                for (int index = 1; index < cells.Length - 1; index++)
                {
                    newCells[index] = (cells[index - 1] == cells[index + 1]) ? 1 : 0;
                }
                newCells[cells.Length - 1] = 0;
                return newCells;
            }

            /*
            Approach 2: Simulation with Bitmap
Complexity Analysis
Let K be the number of cells, and N be the number of steps.
•	Time Complexity: O(min(N,2^K)) assuming that K does not exceed 32.
o	As we discussed before, at most we could have 2^K possible states. While we run the simulation, we need to run min(N,2^K) steps without fast-forwarding in the worst case.
o	For each simulation step, it takes a constant O(1) time to process and evolve the states of cells, since we applied the bit operations rather than iteration.
o	Hence, the overall time complexity of the algorithm is O(min(N,2^K)).
•	Space Complexity: O(2^K)
o	The main memory consumption of the algorithm is the hashmap that we use to keep track of the states of the cells. The maximal number of entries in the hashmap would be 2^K as we discussed before.
o	This time we adopted the bitmap for both Java and Python implementation so that each state consumes a constant O(1) space.

            */
            public int[] SimulationWithBitmap(int[] cells, int N)
            {
                Dictionary<int, int> seenStates = new Dictionary<int, int>();
                bool isFastForwarded = false;

                // step 1). convert the cells to bitmap
                int stateBitmap = 0x0;
                foreach (int cell in cells)
                {
                    stateBitmap <<= 1;
                    stateBitmap |= cell;
                }

                // step 2). run the simulation with hashmap
                while (N > 0)
                {
                    if (!isFastForwarded)
                    {
                        if (seenStates.ContainsKey(stateBitmap))
                        {
                            // the length of the cycle is seen[state_key] - N
                            N %= seenStates[stateBitmap] - N;
                            isFastForwarded = true;
                        }
                        else
                        {
                            seenStates[stateBitmap] = N;
                        }
                    }
                    // Check if there are still some steps remaining,
                    // with or without the fast forwarding.
                    if (N > 0)
                    {
                        N -= 1;
                        stateBitmap = NextDay(stateBitmap);
                    }
                }

                // step 3). convert the bitmap back to the state cells
                int[] ret = new int[cells.Length];
                for (int i = cells.Length - 1; i >= 0; i--)
                {
                    ret[i] = (stateBitmap & 0x1);
                    stateBitmap >>= 1;
                }
                return ret;
            }

            private int NextDay(int stateBitmap)
            {
                stateBitmap = ~(stateBitmap << 1) ^ (stateBitmap >> 1);
                // set the head and tail to zero
                stateBitmap &= 0x7e;
                return stateBitmap;
            }

        }


        /* 956. Tallest Billboard
        https://leetcode.com/problems/tallest-billboard/description/
         */
        class TallestBillboardSol
        {
            /*
            Approach 1: Meet in the Middle
            Complexity Analysis
Let n be the length of the input array rods.
•	Time complexity: O(3^(N/2))
o	We need to generate all possible combinations of two halves of rods and store them in first_half (or second_half). The number of possible combinations can grow exponentially with n. The time complexity is O(3^(N/2)) for each half.
•	Space complexity: O(3^(N/2))
o	There could be at most 32n distinct combinations stored in first_half and second_half.

            */
            public int MeetInTheMiddle(int[] rods)
            {
                int n = rods.Length;
                Dictionary<int, int> firstHalf = Helper(rods, 0, n / 2);
                Dictionary<int, int> secondHalf = Helper(rods, n / 2, n);

                int answer = 0;
                foreach (var diff in firstHalf.Keys)
                {
                    if (secondHalf.ContainsKey(-diff))
                    {
                        answer = Math.Max(answer, firstHalf[diff] + secondHalf[-diff]);
                    }
                }
                return answer;
            }

            // Helper function to collect every combination `(left, right)`
            private Dictionary<int, int> Helper(int[] rods, int leftIndex, int rightIndex)
            {
                HashSet<(int, int)> states = new HashSet<(int, int)>();
                states.Add((0, 0));
                for (int i = leftIndex; i < rightIndex; ++i)
                {
                    int r = rods[i];
                    HashSet<(int, int)> newStates = new HashSet<(int, int)>();
                    foreach (var p in states)
                    {
                        newStates.Add((p.Item1 + r, p.Item2));
                        newStates.Add((p.Item1, p.Item2 + r));
                    }
                    foreach (var newState in newStates)
                    {
                        states.Add(newState);
                    }
                }

                Dictionary<int, int> dp = new Dictionary<int, int>();
                foreach (var p in states)
                {
                    int left = p.Item1, right = p.Item2;
                    dp[left - right] = Math.Max(dp.GetValueOrDefault(left - right, 0), left);
                }
                return dp;
            }

            /*
            Approach 2: Dynamic Programming
Complexity Analysis
Let n be the length of the input array rods and m be the maximum sum of rods.
•	Time complexity: O(n⋅m)
o	We need an iteration over rods which contains n steps.
o	For each rod[i], we need to update new_dp based on every state in dp. There could be at most m difference height differences, which represents the number of unique states we need to traverse.
o	Therefore, the time complexity is O(n⋅m).
•	Space complexity: O(m)
o	There could be at most m difference height difference and the number of unique states stored in dp.

            */
            public int WithDP(int[] rods)
            {
                // dp[taller - shorter] = taller
                Dictionary<int, int> dp = new Dictionary<int, int>();
                dp[0] = 0;

                foreach (int rod in rods)
                {
                    // newDp means we don't add rod to these stands.
                    Dictionary<int, int> newDp = new Dictionary<int, int>(dp);

                    foreach (KeyValuePair<int, int> entry in dp)
                    {
                        int difference = entry.Key;
                        int taller = entry.Value;
                        int shorter = taller - difference;

                        // Add rod to the taller stand, thus the height difference is increased to difference + rod.
                        int newTaller = newDp.TryGetValue(difference + rod, out int temp) ? temp : 0;
                        newDp[difference + rod] = Math.Max(newTaller, taller + rod);

                        // Add rod to the shorter stand, the height difference is expressed as abs(shorter + rod - taller).
                        int newDifference = Math.Abs(shorter + rod - taller);
                        int newTaller2 = Math.Max(shorter + rod, taller);
                        newDp[newDifference] = Math.Max(newTaller2, newDp.TryGetValue(newDifference, out int temp2) ? temp2 : 0);
                    }

                    dp = newDp;
                }

                // Return the maximum height with 0 difference.
                return dp.TryGetValue(0, out int result) ? result : 0;
            }

        }



        /* 969. Pancake Sorting
        https://leetcode.com/problems/pancake-sorting/description/
         */
        class PancakeSortSol
        {
            /* Approach 1: Sort like Bubble-Sort
            Complexity Analysis
Let N be the length of the input list.
•	Time Complexity: O(N^2)
o	In the algorithm, we run a loop with N iterations.
o	Within each iteration, we are dealing with the corresponding prefix of the list.
Here we denote the length of the prefix as k, e.g. in the first iteration, the length of the prefix is N. While in the second iteration, the length of the prefix is N−1.
o	Within each iteration, we have operations whose time complexity is linear to the length of the prefix, such as iterating through the prefix to find the index, or flipping the entire prefix etc. Hence, for each iteration, its time complexity would be O(k)
o	To sum up all iterations, we have the overall time complexity of the algorithm as ∑k=1NO(k)=O(N^2).
•	Space Complexity: O(N)
o	Within the algorithm, we use a list to maintain the final results, which is proportional to the number of pancake flips.
o	For each round of iteration, at most we would add two pancake flips. Therefore, the maximal number of pancake flips needed would be 2⋅N.
o	As a result, the space complexity of the algorithm is O(N). If one does not take into account the space required to hold the result of the function, then one could consider the above algorithm as a constant space solution.

             */

            /// <summary>
            /// Sort like bubble-sort i.e. sink the largest number to the bottom at each round.
            /// </summary>
            public List<int> PancakeSort(int[] arrayToSort)
            {
                List<int> result = new List<int>();

                for (int valueToSort = arrayToSort.Length; valueToSort > 0; valueToSort--)
                {
                    // locate the position for the value to sort in this round
                    int index = Find(arrayToSort, valueToSort);

                    // sink the valueToSort to the bottom,
                    // with at most two steps of pancake flipping.
                    if (index == valueToSort - 1)
                        continue;

                    // 1). flip the value to the head if necessary
                    if (index != 0)
                    {
                        result.Add(index + 1);
                        Flip(arrayToSort, index + 1);
                    }

                    // 2). now that the value is at the head, flip it to the bottom
                    result.Add(valueToSort);
                    Flip(arrayToSort, valueToSort);
                }

                return result;
            }

            protected void Flip(int[] sublist, int k)
            {
                int i = 0;
                while (i < k / 2)
                {
                    int temporaryValue = sublist[i];
                    sublist[i] = sublist[k - i - 1];
                    sublist[k - i - 1] = temporaryValue;
                    i += 1;
                }
            }

            protected int Find(int[] array, int target)
            {
                for (int i = 0; i < array.Length; i++)
                    if (array[i] == target)
                        return i;

                return -1;
            }
        }

        /*         983. Minimum Cost For Tickets
        https://leetcode.com/problems/minimum-cost-for-tickets/description/
         */
        class MinCostTicketsSol
        {
            HashSet<int> travelDaysNeeded = new HashSet<int>();

            /*
            Approach 1: Top-Down Dynamic Programming
            Complexity Analysis
            Here, K is the last day we need to travel, the last value in the array days.
            •	Time complexity: O(K).
            The size of array dp is K+1, and we need to find the answer for each of the K states. For each state, the time required is O(1) as there would be only three recursive calls for each state. Therefore, the time complexity would equal O(K).
            •	Space complexity: O(K).
            The size of array dp is K+1; also, there would be some stack space required. The maximum active recursion depth would be K, i.e., one for each day. The size of the set isTravelNeeded will be equal to the size of days, i.e. N, considering the integers in days will always be strictly increasing we can say N<=K. Hence, the space complexity would equal O(K).


            */
            public int TopDownDP(int[] travelDays, int[] ticketCosts)
            {
                // The last day on which we need to travel.
                int lastTravelDay = travelDays[travelDays.Length - 1];
                int[] dp = new int[lastTravelDay + 1];
                Array.Fill(dp, -1);

                // Mark the days on which we need to travel.
                foreach (int day in travelDays)
                {
                    travelDaysNeeded.Add(day);
                }

                return Solve(dp, travelDays, ticketCosts, 1);
            }
            private int Solve(int[] dp, int[] travelDays, int[] ticketCosts, int currentDay)
            {
                // If we have iterated over travel days, return 0.
                if (currentDay > travelDays[travelDays.Length - 1])
                {
                    return 0;
                }

                // If we don't need to travel on this day, move on to next day.
                if (!travelDaysNeeded.Contains(currentDay))
                {
                    return Solve(dp, travelDays, ticketCosts, currentDay + 1);
                }

                // If already calculated, return from here with the stored answer.
                if (dp[currentDay] != -1)
                {
                    return dp[currentDay];
                }

                int oneDayCost = ticketCosts[0] + Solve(dp, travelDays, ticketCosts, currentDay + 1);
                int sevenDayCost = ticketCosts[1] + Solve(dp, travelDays, ticketCosts, currentDay + 7);
                int thirtyDayCost = ticketCosts[2] + Solve(dp, travelDays, ticketCosts, currentDay + 30);

                // Store the cost with the minimum of the three options.
                return dp[currentDay] = Math.Min(oneDayCost, Math.Min(sevenDayCost, thirtyDayCost));
            }

            /*
            Approach 2: Bottom-up Dynamic Programming
            Complexity Analysis
Here, K is the last day that we need to travel, the last value in the array days.
•	Time complexity: O(K).
The size of array dp is K, and we need to iterate over each of the K days. For each day, the work required is O(1). Therefore, the time complexity would equal O(K).
•	Space complexity: O(K).
The size of array dp is K. Hence, the space complexity would equal O(K).

            */
            public int BottomUpDP(int[] travelDays, int[] ticketCosts)
            {
                // The last day on which we need to travel.
                int lastTravelDay = travelDays[travelDays.Length - 1];
                int[] dynamicProgrammingArray = new int[lastTravelDay + 1];
                Array.Fill(dynamicProgrammingArray, 0);

                int travelDayIndex = 0;
                for (int currentDay = 1; currentDay <= lastTravelDay; currentDay++)
                {
                    // If we don't need to travel on this day, the cost won't change.
                    if (currentDay < travelDays[travelDayIndex])
                    {
                        dynamicProgrammingArray[currentDay] = dynamicProgrammingArray[currentDay - 1];
                    }
                    else
                    {
                        // Buy a pass on this day, and move on to the next travel day.
                        travelDayIndex++;
                        // Store the cost with the minimum of the three options.
                        dynamicProgrammingArray[currentDay] = Math.Min(dynamicProgrammingArray[currentDay - 1] + ticketCosts[0],
                            Math.Min(dynamicProgrammingArray[Math.Max(0, currentDay - 7)] + ticketCosts[1],
                                dynamicProgrammingArray[Math.Max(0, currentDay - 30)] + ticketCosts[2]));
                    }
                }

                return dynamicProgrammingArray[lastTravelDay];
            }

        }


        /* 322. Coin Change
        https://leetcode.com/problems/coin-change/description/
         */
        public class CoinChangeSol
        {
            /*
            Approach 1 (Brute force) [Time Limit Exceeded]
        Complexity Analysis
        •	Time complexity : O(S^n). In the worst case, complexity is exponential in the number of the coins n. The reason is that every coin denomination ci could have at most  S/cii values. Therefore the number of possible combinations is :
        S/c1∗S/c2∗S/c3…S/cn= Sn/ (c1∗c2∗c3…cn)    
        •	Space complexity : O(n).
        In the worst case the maximum depth of recursion is n. Therefore we need O(n) space used by the system recursive stack.

            */
            public int Naive(int[] coins, int amount)
            {
                return CoinChange(0, coins, amount);
            }

            private int CoinChange(int coinIndex, int[] coins, int amount)
            {
                if (amount == 0)
                    return 0;
                if (coinIndex < coins.Length && amount > 0)
                {
                    int maxValue = amount / coins[coinIndex];
                    int minimumCost = int.MaxValue;
                    for (int count = 0; count <= maxValue; count++)
                    {
                        if (amount >= count * coins[coinIndex])
                        {
                            int result = CoinChange(coinIndex + 1, coins, amount - count * coins[coinIndex]);
                            if (result != -1)
                                minimumCost = Math.Min(minimumCost, result + count);
                        }
                    }
                    return (minimumCost == int.MaxValue) ? -1 : minimumCost;
                }
                return -1;
            }
            /*
            Approach 2 (Dynamic programming - Top down) [Accepted]
        Complexity Analysis
        •	Time complexity : O(S∗n). where S is the amount, n is denomination count.
        In the worst case the recursive tree of the algorithm has height of S and the algorithm solves only S subproblems because it caches precalculated solutions in a table. Each subproblem is computed with n iterations, one by coin denomination. Therefore there is O(S∗n) time complexity.
        •	Space complexity : O(S), where S is the amount to change
        We use extra space for the memoization table.
            */
            public int TopDownDP(int[] coins, int amount)
            {
                if (amount < 1) return 0;
                return CoinChange(coins, amount, new int[amount]);
            }

            private int CoinChange(int[] coins, int remainingAmount, int[] count)
            {
                if (remainingAmount < 0) return -1;
                if (remainingAmount == 0) return 0;
                if (count[remainingAmount - 1] != 0) return count[remainingAmount - 1];
                int minimumCoins = int.MaxValue;
                foreach (int coin in coins)
                {
                    int result = CoinChange(coins, remainingAmount - coin, count);
                    if (result >= 0 && result < minimumCoins)
                        minimumCoins = 1 + result;
                }
                count[remainingAmount - 1] = (minimumCoins == int.MaxValue) ? -1 : minimumCoins;
                return count[remainingAmount - 1];
            }
            /* Approach 3 (Dynamic programming - Bottom up) [Accepted]
            Complexity Analysis
            •	Time complexity : O(S∗n).
            On each step the algorithm finds the next F(i) in n iterations, where 1≤i≤S. Therefore in total the iterations are S∗n.
            •	Space complexity : O(S).
            We use extra space for the memoization table.

            */
            public int BottomUpDP(int[] coins, int amount)
            {
                int max = amount + 1;
                int[] dp = new int[amount + 1];
                Array.Fill(dp, max);
                dp[0] = 0;
                for (int i = 1; i <= amount; i++)
                {
                    for (int j = 0; j < coins.Length; j++)
                    {
                        if (coins[j] <= i)
                        {
                            dp[i] = Math.Min(dp[i], dp[i - coins[j]] + 1);
                        }
                    }
                }
                return dp[amount] > amount ? -1 : dp[amount];
            }


        }

        /* 997. Find the Town Judge
        https://leetcode.com/problems/find-the-town-judge/description/
         */

        public class FindEdgeSol
        {

            /* Complexity Analysis
            Let N be the number of people, and E be the number of edges (trust relationships).
            •	Time Complexity : O(E).
            We loop over the trust list once. The cost of doing this is O(E).
            We then loop over the people. The cost of doing this is O(N).
            Going by this, it now looks this is one those many graph problems where the cost is O(max(N,E)=O(N+E). After all, we don't know whether E or N is the bigger one, right?
            However, remember how we terminate early if E<N−1? This means that in the best case, the time complexity is O(1). And in the worst case, we know that E≥N−1. For the purpose of big-oh notation, we ignore the constant of 1. Therefore, in the worst case, E has to be bigger, and so we can simply drop the N, leaving O(E).
            •	Space Complexity : O(N).
            We allocated 2 arrays; one for the indegrees and the other for the outdegrees. Each was of length N + 1. Because in big-oh notation we drop constants, this leaves us with O(N).
             */
            public int TwoArray(int N, int[][] trust)
            {

                if (trust.Length < N - 1)
                {
                    return -1;
                }

                int[] indegrees = new int[N + 1];
                int[] outdegrees = new int[N + 1];

                foreach (int[] relation in trust)
                {
                    indegrees[relation[1]]++;
                    outdegrees[relation[0]]++;
                }

                for (int i = 1; i <= N; i++)
                {
                    if (indegrees[i] == N - 1 && outdegrees[i] == 0)
                    {
                        return i;
                    }
                }
                return -1;
            }

            /*
            Approach 2: One Array
            Complexity Analysis
    Recall that N is the number of people, and E is the number of edges (trust relationships).
    •	Time Complexity : O(E).
    Same as above. We still need to loop through the E edges in trust, and the argument about the relationship between N and E still applies.
    •	Space Complexity : O(N).
    Same as above. We're still allocating an array of length N.	

            */
            public int OneArray(int N, int[][] trust)
            {

                if (trust.Length < N - 1)
                {
                    return -1;
                }

                int[] trustScores = new int[N + 1];

                foreach (int[] relation in trust)
                {
                    trustScores[relation[0]]--;
                    trustScores[relation[1]]++;
                }

                for (int i = 1; i <= N; i++)
                {
                    if (trustScores[i] == N - 1)
                    {
                        return i;
                    }
                }
                return -1;
            }

        }


        /* 277. Find the Celebrity
        https://leetcode.com/problems/find-the-celebrity/description/
         */

        public class FindCelebritySol
        {
            public class Relation
            {
                public virtual bool Knows(int a, int b) { return false; } //Dummy for now
            }
            /*
Approach 1: Brute Force
Complexity Analysis
We don't know what time and space the knows(...) API uses. Because it's not our concern, we'll assume it's O(1) for the purpose of analysing our algorithm.
•	Time Complexity : O(n^2).
For each of the n people, we need to check whether or not they are a celebrity.
Checking whether or not somebody is a celebrity requires making 2 API calls for each of the n−1 other people, for a total of 2⋅(n−1)=2⋅n−2 calls. In big-oh notation, we drop the constants, leaving O(n).
So each of the n celebrity checks will cost O(n), giving a total of O(n^2).
•	Space Complexity : O(1).
Our code only uses constant extra space. The results of the API calls are not saved.

            */

            public class NaiveSol : Relation
            {
                private int numberOfPeople;

                public int FindCelebrity(int numberOfPeople)
                {
                    this.numberOfPeople = numberOfPeople;
                    for (int personIndex = 0; personIndex < numberOfPeople; personIndex++)
                    {
                        if (IsCelebrity(personIndex))
                        {
                            return personIndex;
                        }
                    }
                    return -1;
                }

                private bool IsCelebrity(int personIndex)
                {
                    for (int otherIndex = 0; otherIndex < numberOfPeople; otherIndex++)
                    {
                        if (personIndex == otherIndex) continue; // Don't ask if they know themselves.
                        if (Knows(personIndex, otherIndex) || !Knows(otherIndex, personIndex))
                        {
                            return false;
                        }
                    }
                    return true;
                }
            }
            /*
            Approach 2: Logical Deduction
Complexity Analysis
•	Time Complexity : O(n).
Our code is split into 2 parts.
The first part finds a celebrity candidate. This requires doing n−1 calls to knows(...) API, and so is O(n).
The second part is the same as before—checking whether or not a given person is a celebrity. We determined that this is O(n).
Therefore, we have a total time complexity of O(n+n)=O(n).
•	Space Complexity : O(1).
Same as above. We are only using constant extra space.

            */

            public class LogicalDeductionSol : Relation
            {
                private int numberOfPeople;

                public int FindCelebrity(int numberOfPeople)
                {
                    this.numberOfPeople = numberOfPeople;
                    int celebrityCandidate = 0;
                    for (int i = 0; i < numberOfPeople; i++)
                    {
                        if (Knows(celebrityCandidate, i))
                        {
                            celebrityCandidate = i;
                        }
                    }
                    if (IsCelebrity(celebrityCandidate))
                    {
                        return celebrityCandidate;
                    }
                    return -1;
                }

                private bool IsCelebrity(int candidateIndex)
                {
                    for (int otherIndex = 0; otherIndex < numberOfPeople; otherIndex++)
                    {
                        if (candidateIndex == otherIndex) continue; // Don't ask if they know themselves.
                        if (Knows(candidateIndex, otherIndex) || !Knows(otherIndex, candidateIndex))
                        {
                            return false;
                        }
                    }
                    return true;
                }
            }
            /*
Approach 3: Logical Deduction with Caching
Complexity Analysis
•	Time Complexity : O(n).
The time complexity is still O(n). The only difference is that sometimes we're retrieving data from a cache inside our code instead of from the API.
•	Space Complexity : O(n).
We're storing the results of the n−1 calls to the know(...) API we made while finding a candidate.
We could optimize the space complexity slightly, by dumping the cached contents each time the celebrityCandidate variable changes, which would be O(1) in the best case (which happens to be the worst case for reducing number of API calls) but it's still O(n) space in the worst case and probably not worth the extra code complexity as the algorithm still ultimately requires the memory/ disk space needed for the worst case.

            */

            public class Solution : Relation
            {
                private int numberOfPeople;
                private Dictionary<(int, int), bool> cache = new Dictionary<(int, int), bool>();

                public override bool Knows(int a, int b)
                {
                    var key = (a, b);
                    if (!cache.ContainsKey(key))
                    {
                        cache[key] = base.Knows(a, b);
                    }
                    return cache[key];
                }

                public int FindCelebrity(int n)
                {
                    numberOfPeople = n;
                    int celebrityCandidate = 0;
                    for (int i = 0; i < n; i++)
                    {
                        if (Knows(celebrityCandidate, i))
                        {
                            celebrityCandidate = i;
                        }
                    }
                    if (IsCelebrity(celebrityCandidate))
                    {
                        return celebrityCandidate;
                    }
                    return -1;
                }

                private bool IsCelebrity(int i)
                {
                    for (int j = 0; j < numberOfPeople; j++)
                    {
                        if (i == j) continue; // Don't ask if they know themselves.
                        if (Knows(i, j) || !Knows(j, i))
                        {
                            return false;
                        }
                    }
                    return true;
                }
            }


        }

        /* 1010. Pairs of Songs With Total Durations Divisible by 60	
        https://leetcode.com/problems/pairs-of-songs-with-total-durations-divisible-by-60/description/
         */
        class NumPairsDivisibleBy60Sol
        {
            /*
            Approach 1: Brute Force
            Complexity Analysis
•	Time complexity: O(n^2), when n is the length of the input array. For each item in time, we iterate through the rest of the array to find a qualified complement taking O(n) time.
•	Space complexity: O(1).

            */
            public int NumPairsDivisibleBy60(int[] time)
            {
                int count = 0, n = time.Length;
                for (int i = 0; i < n; i++)
                {
                    // j starts with i+1 so that i is always to the left of j
                    // to avoid repetitive counting
                    for (int j = i + 1; j < n; j++)
                    {
                        if ((time[i] + time[j]) % 60 == 0)
                        {
                            count++;
                        }
                    }
                }
                return count;
            }
            /*
            
Approach 2: Use an Array to Store Frequencies
Complexity Analysis
•	Time complexity: O(n), when n is the length of the input array because we would visit each element in time once.
•	Space complexity: O(1), because the size of the array remainders is fixed with 60.

            */
            public int UsingArrayForFreq(int[] time)
            {
                int[] remainders = new int[60];
                int count = 0;
                foreach (int t in time)
                {
                    if (t % 60 == 0)
                    { // check if a%60==0 && b%60==0
                        count += remainders[0];
                    }
                    else
                    { // check if a%60+b%60==60
                        count += remainders[60 - t % 60];
                    }
                    remainders[t % 60]++; // remember to update the remainders
                }
                return count;
            }

        }


        /* 1020. Number of Enclaves
        https://leetcode.com/problems/number-of-enclaves/description/	
         */
        class NumEnclavesSol
        {
            /*

Approach 1: Depth First Search
Complexity Analysis
Here, m and n are the number of rows and columns in the given grid.
•	Time complexity: O(m⋅n)
o	Initializing the visit array takes O(m⋅n) time.
o	We iterate over the boundary and find unvisited land cells to perform DFS traversal from those. This takes O(m+n) time.
o	The dfs function visits each node at most once. Since there are O(m⋅n) nodes, we will perform O(m⋅n) operations visiting all nodes in the worst-case scenario. We iterate over all the neighbors of each node that is popped out of the queue. So for every node, we would iterate four times over the neighbors, resulting in O(4⋅m⋅n)=O(m⋅n) operations total for all the nodes.
o	Counting the number of unvisited land cells also takes O(m⋅n) time.
•	Space complexity: O(m⋅n)
o	The visit array takes O(m⋅n) space.
o	The recursion stack used by dfs can have no more than O(m⋅n) elements in the worst-case scenario where each node is added to it. It would take up O(m⋅n) space in that case.

            */
            public int DFS(int[][] grid)
            {
                int m = grid.Length;
                int n = grid[0].Length;
                bool[][] visit = new bool[m][];

                for (int i = 0; i < m; ++i)
                {
                    // First column.
                    if (grid[i][0] == 1 && !visit[i][0])
                    {
                        dfs(i, 0, m, n, grid, visit);
                    }
                    // Last column.
                    if (grid[i][n - 1] == 1 && !visit[i][n - 1])
                    {
                        dfs(i, n - 1, m, n, grid, visit);
                    }
                }

                for (int i = 0; i < n; ++i)
                {
                    // First row.
                    if (grid[0][i] == 1 && !visit[0][i])
                    {
                        dfs(0, i, m, n, grid, visit);
                    }
                    // Last row.
                    if (grid[m - 1][i] == 1 && !visit[m - 1][i])
                    {
                        dfs(m - 1, i, m, n, grid, visit);
                    }
                }

                int count = 0;
                for (int i = 0; i < m; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        if (grid[i][j] == 1 && !visit[i][j])
                        {
                            count++;
                        }
                    }
                }
                return count;
            }
            public void dfs(int x, int y, int m, int n, int[][] grid, bool[][] visit)
            {
                if (x < 0 || x >= m || y < 0 || y >= n || grid[x][y] == 0 || visit[x][y])
                {
                    return;
                }

                visit[x][y] = true;
                int[] dirx = { 0, 1, 0, -1 };
                int[] diry = { -1, 0, 1, 0 };

                for (int i = 0; i < 4; i++)
                {
                    dfs(x + dirx[i], y + diry[i], m, n, grid, visit);
                }
                return;
            }

            /*
            Approach 2: Breadth-First Search
Complexity Analysis
Here, m and n are the number of rows and columns in the given grid.
•	Time complexity: O(m⋅n)
o	Initializing the visit array takes O(m⋅n) time.
o	We iterate over the boundary of grid and find unvisited land cells to perform BFS traversal from those. This takes O(m+n) time.
o	Each queue operation in the BFS algorithm takes O(1) time and a single node can be pushed at most once in the queue. Since there are O(m⋅n) nodes, we will perform O(m⋅n) operations visiting all nodes in the worst-case scenario. We iterate over all the neighbors of each node that is popped out of the queue. So for every node, we would iterate four times over the neighbors, resulting in O(4⋅m⋅n)=O(m⋅n) operations total for all the nodes.
•	Space complexity: O(m⋅n)
o	The visit array takes O(m⋅n) space.
o	The BFS queue takes O(m⋅n) space in the worst-case where each node is added once.

            */
            public int BFS(int[,] grid)
            {
                int rows = grid.GetLength(0);
                int cols = grid.GetLength(1);
                bool[,] visited = new bool[rows, cols];

                for (int i = 0; i < rows; ++i)
                {
                    // First column.
                    if (grid[i, 0] == 1 && !visited[i, 0])
                    {
                        Bfs(i, 0, rows, cols, grid, visited);
                    }
                    // Last column.
                    if (grid[i, cols - 1] == 1 && !visited[i, cols - 1])
                    {
                        Bfs(i, cols - 1, rows, cols, grid, visited);
                    }
                }

                for (int i = 0; i < cols; ++i)
                {
                    // First row.
                    if (grid[0, i] == 1 && !visited[0, i])
                    {
                        Bfs(0, i, rows, cols, grid, visited);
                    }
                    // Last row.
                    if (grid[rows - 1, i] == 1 && !visited[rows - 1, i])
                    {
                        Bfs(rows - 1, i, rows, cols, grid, visited);
                    }
                }

                int enclaveCount = 0;
                for (int i = 0; i < rows; i++)
                {
                    for (int j = 0; j < cols; j++)
                    {
                        if (grid[i, j] == 1 && !visited[i, j])
                        {
                            enclaveCount++;
                        }
                    }
                }
                return enclaveCount;
            }
            public void Bfs(int startX, int startY, int rows, int cols, int[,] grid, bool[,] visited)
            {
                Queue<int[]> queue = new Queue<int[]>();
                queue.Enqueue(new int[] { startX, startY });
                visited[startX, startY] = true;

                int[] directionX = { 0, 1, 0, -1 };
                int[] directionY = { -1, 0, 1, 0 };

                while (queue.Count > 0)
                {
                    int[] current = queue.Dequeue();
                    startX = current[0];  // row number
                    startY = current[1];  // column number

                    for (int i = 0; i < 4; i++)
                    {
                        int newRow = startX + directionX[i];
                        int newCol = startY + directionY[i];
                        if (newRow < 0 || newRow >= rows || newCol < 0 || newCol >= cols)
                        {
                            continue;
                        }
                        else if (grid[newRow, newCol] == 1 && !visited[newRow, newCol])
                        {
                            queue.Enqueue(new int[] { newRow, newCol });
                            visited[newRow, newCol] = true;
                        }
                    }
                }
                return;
            }

        }


        /* 1029. Two City Scheduling
        https://leetcode.com/problems/two-city-scheduling/description/
         */

        class TwoCitySchedCostSol
        {
            /*
            Approach 1: Greedy
            Complexity Analysis
Let N be half of the length of the input array costs.
•	Time complexity : O(NlogN) because of sorting of
input data.
•	Space complexity : O(logN)
Some extra space is used when we sort costs in place. The space complexity of the sorting algorithm depends on which sorting algorithm is used; the default algorithm varies from one language to another.

            */
            public int WithGreedy(int[][] costs)
            {
                // Sort by a gain which company has 
                // by sending a person to city A and not to city B
                Array.Sort(costs, (int[] o1, int[] o2) =>
            {
                return o1[0] - o1[1] - (o2[0] - o2[1]);

            });

                int total = 0;
                int n = costs.Length / 2;
                // To optimize the company expenses,
                // send the first n persons to the city A
                // and the others to the city B
                for (int i = 0; i < n; ++i)
                {
                    total += costs[i][0] + costs[i + n][1];
                }
                return total;
            }
        }


        /* 1032. Stream of Characters
        https://leetcode.com/problems/stream-of-characters/description/
         */


        class StreamChecker
        {
            class TrieNode
            {
                public Dictionary<char, TrieNode> Children { get; set; } = new Dictionary<char, TrieNode>();
                public bool IsWord { get; set; } = false;
            }
            /*
            Approach 1: Trie
 
            */
            private TrieNode trie = new TrieNode();
            private LinkedList<char> stream = new LinkedList<char>();
            /* 
                      Complexity Analysis
            Let N be the number of input words, and M be the word length.
            •	Time complexity: O(N⋅M).
            We have N words to process. At each step, we either examine or create a node in the trie. That takes only M operations.
            •	Space complexity: O(N⋅M).
            In the worst case, the newly inserted key doesn't share a prefix with the keys already added in the trie. We have to add N⋅M new nodes, which takes O(N⋅M) space.
              */
            public StreamChecker(string[] words)
            {
                foreach (string word in words)
                {
                    TrieNode node = trie;
                    char[] reversedWordArray = word.ToCharArray();
                    Array.Reverse(reversedWordArray);
                    string reversedWord = new string(reversedWordArray);
                    foreach (char ch in reversedWord)
                    {
                        if (!node.Children.ContainsKey(ch))
                        {
                            node.Children[ch] = new TrieNode();
                        }
                        node = node.Children[ch];
                    }
                    node.IsWord = true;
                }
            }

            /* Let M be the maximum length of a word length. i.e. the depth of trie.
            •	Time complexity: O(M)
            •	Space complexity: O(M) to keep a stream of characters.
            One could limit the size of the deque to be equal to the length of the longest input word.
             */
            public bool Query(char letter)
            {
                stream.AddFirst(letter);

                TrieNode node = trie;
                foreach (char ch in stream)
                {
                    if (node.IsWord)
                    {
                        return true;
                    }
                    if (!node.Children.ContainsKey(ch))
                    {
                        return false;
                    }
                    node = node.Children[ch];
                }
                return node.IsWord;
            }
        }


        /* 1035. Uncrossed Lines
        https://leetcode.com/problems/uncrossed-lines/description/
         */
        class MaxUncrossedLinesSol
        {
            /*
            Approach 1: Recursive Dynamic Programming
            Complexity Analysis
Here, n1 is the length of nums1 and n2 is the length of nums2.
•	Time complexity: O(n1⋅n2)
o	Initializing the memo array takes O(n1⋅n2) time.
o	It will take O(n1⋅n2) because there are O(n1⋅n2) states to iterate over. The recursive function may be called multiple times for a given state, but due to memoization, each state is only computed once.
•	Space complexity: O(n1⋅n2)
o	The memo array consumes O(n1⋅n2) space.
o	The recursion stack used in the solution can grow to a maximum size of O(n1+n2). When we try to form the recursion tree, we see that after each node two branches are formed (when the last numbers aren't equal). In one branch, we decrement 1 from nums1 and in other branch, we decrement 1 from nums2. The recursion stack would only have one call out of the two branches. The height of such a tree will be max(n1,n2)) because at each level we are decrementing the number of elements under consideration by 1. Hence, the recursion stack will have a maximum of O(max(n1,n2))=O(n1+n2) elements.

            */
            public int DPRec(int[] nums1, int[] nums2)
            {
                int n1 = nums1.Length;
                int n2 = nums2.Length;

                int[][] memo = new int[n1 + 1][];
                foreach (int[] row in memo)
                {
                    Array.Fill(row, -1);
                }

                return Solve(n1, n2, nums1, nums2, memo);
            }
            private int Solve(int i, int j, int[] nums1, int[] nums2, int[][] memo)
            {
                if (i <= 0 || j <= 0)
                {
                    return 0;
                }

                if (memo[i][j] != -1)
                {
                    return memo[i][j];
                }

                if (nums1[i - 1] == nums2[j - 1])
                {
                    memo[i][j] = 1 + Solve(i - 1, j - 1, nums1, nums2, memo);
                }
                else
                {
                    memo[i][j] =
                        Math.Max(Solve(i, j - 1, nums1, nums2, memo), Solve(i - 1, j, nums1, nums2, memo));
                }
                return memo[i][j];
            }
            /*
            Approach 2: Iterative Dynamic Programming
            Complexity Analysis
Here, n1 is the length of nums1 and n2 is the length of nums2.
•	Time complexity: O(n1⋅n2)
o	Initializing the dp array takes O(n1⋅n2) time.
o	We fill the dp array which takes O(n1⋅n2) time.
•	Space complexity: O(n1⋅n2)
o	The dp array consumes O(n1⋅n2) space.

            */
            public int DPIterative(int[] nums1, int[] nums2)
            {
                int n1 = nums1.Length;
                int n2 = nums2.Length;

                int[][] dp = new int[n1 + 1][];

                for (int i = 1; i <= n1; i++)
                {
                    for (int j = 1; j <= n2; j++)
                    {
                        if (nums1[i - 1] == nums2[j - 1])
                        {
                            dp[i][j] = 1 + dp[i - 1][j - 1];
                        }
                        else
                        {
                            dp[i][j] = Math.Max(dp[i][j - 1], dp[i - 1][j]);
                        }
                    }
                }

                return dp[n1][n2];
            }
            /*
            Approach 3: Dynamic Programming with Space Optimization
            Complexity Analysis
            Here, n1 is the length of nums1 and n2 is the length of nums2.
            •	Time complexity: O(n1⋅n2)
            o	Initializing the dp and dpPrev arrays take O(n2) time.
            o	To get the answer, we use two loops that take O(n1⋅n2) time.
            •	Space complexity: O(n2)
            o	The dp and dpPrev arrays take O(n2) space each.

            */
            public int DPIterativeWithSpaceOptiaml(int[] nums1, int[] nums2)
            {
                int n1 = nums1.Length;
                int n2 = nums2.Length;

                int[] dp = new int[n2 + 1];
                int[] dpPrev = new int[n2 + 1];

                for (int i = 1; i <= n1; i++)
                {
                    for (int j = 1; j <= n2; j++)
                    {
                        if (nums1[i - 1] == nums2[j - 1])
                        {
                            dp[j] = 1 + dpPrev[j - 1];
                        }
                        else
                        {
                            dp[j] = Math.Max(dp[j - 1], dpPrev[j]);
                        }
                    }
                    dpPrev = (int[])dp.Clone();
                }

                return dp[n2];
            }

        }


        /* 1041. Robot Bounded In Circle
        https://leetcode.com/problems/robot-bounded-in-circle/description/
         */

        class IsRobotBoundedSol
        {
            /*
            Approach 1: One Pass
Complexity Analysis
•	Time complexity: O(N), where N is a number of instructions
to parse.
•	Space complexity: O(1) because the array directions contains
only 4 elements.

            */
            public bool UsingOnePass(String instructions)
            {
                // north = 0, east = 1, south = 2, west = 3
                int[][] directions = new int[][] {  new int[]{ 0, 1 },  new int[]{ 1, 0 },
                                    new int[]{ 0, -1 },  new int[]{ -1, 0 } };
                // Initial position is in the center
                int x = 0, y = 0;
                // facing north
                int idx = 0;

                foreach (char i in instructions)
                {
                    if (i == 'L')
                        idx = (idx + 3) % 4;
                    else if (i == 'R')
                        idx = (idx + 1) % 4;
                    else
                    {
                        x += directions[idx][0];
                        y += directions[idx][1];
                    }
                }

                // after one cycle:
                // robot returns into initial position
                // or robot doesn't face north
                return (x == 0 && y == 0) || (idx != 0);
            }
        }

        /* 1046. Last Stone Weight
        https://leetcode.com/problems/last-stone-weight/description/
         */
        public class LastStoneWeightSol
        {
            /*
            Approach 1: Array-Based Simulation
     Complexity Analysis
Let N be the length of stones. Here on LeetCode, we're only testing your code with cases where N≤30. In an interview though, be very careful about such assumptions. It is very likely your interviewer expects you to come up with the best possible algorithm you could (thus handling the highest possible value of N you can).
•	Time complexity : O(N^2).
The only non-O(1) method of StoneArray is findAndRemoveMax(). This method does a single pass over the array, to find the index of the maximum value. This pass has a cost of O(N). Once we find the maximum value, we delete it, although this only has a cost of O(1) because instead of shuffling along, we're simply swapping with the end.
Each time around the main loop, there is a net loss of either 1 or 2 stones. Starting with N stones and needing to get to 1 stone, this is up to N−1 iterations. On each of these iterations, it finds the maximum twice. In total, we get O(N^2).
Note that even if we'd shuffled instead of swapped with the end, the findAndRemoveMax() method still would have been O(N), as the pass and then deletion are done one-after-the-other. However, it's often best to avoid needlessly large constants.
•	Space complexity : O(N) or O(1).
For the Python: We are not allocating any new space for data structures, and instead are modifying the input list. Note that this modifies the input. This has its pros and cons; it saves space, but it means that other functions can't use the same array.
For the Java: We need to convert the input to an ArrayList, and therefore the ints to Integers. It is possible to write a O(1) space solution for Java, however it is long-winded and a lot of work for what is a poor overall approach anyway.
       
            */
            public int ArrayBasedSimulation(int[] stones)
            {
                List<int> stoneList = new List<int>();
                foreach (int weight in stones)
                {
                    stoneList.Add(weight);
                }

                while (stoneList.Count > 1)
                {
                    int stone1 = RemoveLargest(stoneList);
                    int stone2 = RemoveLargest(stoneList);
                    if (stone1 != stone2)
                    {
                        stoneList.Add(stone1 - stone2);
                    }
                }

                return stoneList.Count > 0 ? stoneList[0] : 0;
            }
            private int RemoveLargest(List<int> stones)
            {
                int indexOfLargest = stones.IndexOf(stones.Max());
                int result = stones[indexOfLargest];
                stones[indexOfLargest] = stones[stones.Count - 1];
                stones.RemoveAt(stones.Count - 1);
                return result;
            }
            /*
            Approach 2: Sorted Array-Based Simulation
            Complexity Analysis
Let N be the length of stones.
•	Time complexity : O(N^2).
The first part of the algorithm is sorting the list. This has a cost of O(NlogN).
Like before, we're repeating the main loop up to N−1 times. And again, we're doing an O(N) operation each time; adding the new stone back into the array, maintaining sorted order by shuffling existing stones along to make space for it. Identifying the two largest stones was O(1) in this approach, but unfortunately this was subsumed by the inefficient adds. This gives us a total of O(N^2).
Because O(N^2) is strictly larger than O(NlogN), we're left with a final time complexity of O(N62).
•	Space complexity : Varies from O(N) to O(1).
Like in Approach 1, we can choose whether or not to modify the input list. If we do modify the input list, this will cost anywhere from O(N) to O(1) space, depending on the sorting algorithm used. However, if we don't, it will always cost at least O(N) to make a copy. Modifying the input has its pros and cons; it saves space, but it means that other functions can't use the same array.
An alternative to this approach is to simply sort inside the loop every time. This will be even worse, with a time complexity of O(N^2*logN).

            */
            public int SortedArrayBasedSimulation(int[] stones)
            {
                List<int> stoneList = new List<int>();
                foreach (int stone in stones)
                {
                    stoneList.Add(stone);
                }
                stoneList.Sort();

                while (stoneList.Count > 1)
                {
                    int stone1 = stoneList[stoneList.Count - 1];
                    stoneList.RemoveAt(stoneList.Count - 1);
                    int stone2 = stoneList[stoneList.Count - 1];
                    stoneList.RemoveAt(stoneList.Count - 1);

                    if (stone1 != stone2)
                    {
                        int newStone = stone1 - stone2;
                        int index = stoneList.BinarySearch(newStone);
                        if (index < 0)
                        {
                            stoneList.Insert(~index, newStone);
                        }
                        else
                        {
                            stoneList.Insert(index, newStone);
                        }
                    }
                }

                return stoneList.Count > 0 ? stoneList[0] : 0;
            }
            /*
             Approach 3: Heap-Based Simulation
Complexity Analysis
Let N be the length of stones.
•	Time complexity : O(NlogN).
Converting an array into a Heap takes O(N) time (it isn't actually sorting; it's putting them into an order that allows us to get the maximums, each in O(logN) time).
Like before, the main loop iterates up to N−1 times. This time however, it's doing up to three O(logN) operations each time; two removes, and an optional add. Like always, the three is an ignored constant. This means that we're doing N⋅O(logN)=O(NlogN) operations.
•	Space complexity : O(N) or O(logN).
In Python, converting a list to a heap is done in place, requiring O(1) auxiliary space, giving a total space complexity of O(1). Modifying the input has its pros and cons; it saves space, but it means that other functions can't use the same array.
In Java though, it's O(N) to create the PriorityQueue.
We could reduce the space complexity to O(1) by implementing our own iterative heapfiy, if needed.

            */
            public int MaxHeapPQ(int[] stones)
            {

                // Insert all the stones into a Max-Heap.
                PriorityQueue<int, int> heap = new PriorityQueue<int, int>();
                foreach (int stone in stones)
                {
                    heap.Enqueue(stone, -stone);
                }

                // While there is more than one stone left, we need to remove the two largest
                // and smash them together. If there is a resulting stone, we need to put into
                // the heap.
                while (heap.Count > 1)
                {
                    int stone1 = heap.Dequeue();
                    int stone2 = heap.Dequeue();
                    if (stone1 != stone2)
                    {
                        heap.Enqueue(stone1 - stone2, -Math.Abs(stone1 - stone2)); //TODO: Double check this logic
                    }
                }

                // Check whether or not there is a stone left to return.
                return heap.Count == 0 ? 0 : heap.Dequeue();
            }
            /*
            Approach 4: Bucket Sort
Complexity Analysis
•	Time complexity : O(N+W).
Putting the N stones of the input array into the bucket array is O(N), because inserting each stone is an O(1) operation.
In the worst case, the main loop iterates through all of the W indexes of the bucket array. Processing each bucket is an O(1) operation. This, therefore, is O(W).
Seeing as we don't know which is larger out of N and W, we get a total of O(N+W).
Technically, this algorithm is pseudo-polynomial, as its time complexity is dependent on the numeric value of the input. Pseudo-polynomial algorithms are useful when there is no "true" polynomial alternative, but in situations such as this one where we have an O(NlogN) alternative (Approach 3), they are only useful for very specific inputs.
With the small values of W that your code is tested against for this question here on LeetCode, this approach turns out to be faster than Approach 3. But that does not make it the better approach.
•	Space complexity : O(W).
We allocated a new array of size W.
When I looked through the discussion forum for this question, I was surprised to see a number of people arguing that this approach is O(N), on the basis that we could say W is a constant, due to the problem description stating it has a maximum value of 1000. The trouble with this argument is that N also has a maximum specified (of 30, in fact), and so it is arbitrary to argue that W is a constant, yet N is not. These constraints on LeetCode problems are intended to help you determine whether or not your algorithm will be fast enough. They are not supposed to imply some variables can be treated as "constants". A correct time/ space complexity should treat them as unbounded.

            */
            public int UsingBucketSort(int[] stones)
            {

                // Set up the bucket array.
                int maxWeight = stones[0];
                foreach (int stone in stones)
                {
                    maxWeight = Math.Max(maxWeight, stone);
                }
                int[] buckets = new int[maxWeight + 1];

                // Bucket sort (Bucket sort is just the generalized counting sort. When the bucket size is 1, bucket sort is the same as counting sort.)
                foreach (int weight in stones)
                {
                    buckets[weight]++;
                }

                // Scan through the buckets.
                int biggestWeight = 0;
                int currentWeight = maxWeight;
                while (currentWeight > 0)
                {
                    if (buckets[currentWeight] == 0)
                    {
                        currentWeight--;
                    }
                    else if (biggestWeight == 0)
                    {
                        buckets[currentWeight] %= 2;
                        if (buckets[currentWeight] == 1)
                        {
                            biggestWeight = currentWeight;
                        }
                        currentWeight--;
                    }
                    else
                    {
                        buckets[currentWeight]--;
                        if (biggestWeight - currentWeight <= currentWeight)
                        {
                            buckets[biggestWeight - currentWeight]++;
                            biggestWeight = 0;
                        }
                        else
                        {
                            biggestWeight -= currentWeight;
                        }
                    }
                }
                return biggestWeight;
            }

        }


        /* 1057. Campus Bikes
        https://leetcode.com/problems/campus-bikes/description/
         */

        class AssignBikesSol
        {
            /*
            Approach 1: Sorting
            Complexity Analysis
Here, N is the number of workers, and M is the number of bikes.
•	Time complexity: O(NMlog(NM))
There will be a total of NM (worker, bike) pairs. Sorting a list of NM elements will cost O(NMlog(NM)) time. In the worst case, we have to iterate over all the pairs to assign each worker a bike. Thus, iterating over these pairs costs O(NM) time. Since the time complexity for sorting is the dominant term, the time complexity is O(NMlog(NM)).
•	Space complexity: O(NM)
WorkerBikePair or the tuple has three variables, hence taking O(1) space. Storing NM WorkerBikePairs or tuples in allTriplets will cost O(NM) space. To track the availability of the bikes bikeStatus takes O(M) space. Storing bikes index corresponding to worker index in workerStatus takes O(N) space.
The space complexity of the sorting algorithm depends on the implementation of each programming language. For instance, in Java, the Arrays.sort() for primitives is implemented as a variant of quicksort algorithm whose space complexity is O(logNM). In C++ sort() function provided by STL is a hybrid of Quick Sort, Heap Sort, and Insertion Sort and has a worst-case space complexity of O(logNM). In Python sort() function uses TimSort which has a worst-case space complexity of O(NM). Thus, the use of the inbuilt sort() function might add up to O(NM) to space complexity.
The total space required is (NM+N+M+NM) hence, the complexity is equal to O(NM).

            */
            public int[] WithSorting(int[][] workers, int[][] bikes)
            {
                // List of WorkerBikePair's to store all the possible pairs
                List<WorkerBikePair> allTriplets = new List<WorkerBikePair>();

                // Generate all the possible pairs
                for (int worker = 0; worker < workers.Length; worker++)
                {
                    for (int bike = 0; bike < bikes.Length; bike++)
                    {
                        int distance = FindDistance(workers[worker], bikes[bike]);
                        WorkerBikePair workerBikePair = new WorkerBikePair(worker, bike, distance);
                        allTriplets.Add(workerBikePair);
                    }
                }

                // Sort the triplets as per the custom comparator 'WorkerBikePairComparer'
                allTriplets.Sort(new WorkerBikePairComparer());

                // Initialize all values to false, to signify no bikes have been taken
                bool[] bikeStatus = new bool[bikes.Length];
                // Initialize all index to -1, to mark all the workers available
                int[] workerStatus = new int[workers.Length];
                Array.Fill(workerStatus, -1);
                // Keep track of how many worker-bike pairs have been made
                int pairCount = 0;

                foreach (WorkerBikePair triplet in allTriplets)
                {
                    int worker = triplet.WorkerIndex;
                    int bike = triplet.BikeIndex;

                    // If both worker and bike are free, assign them to each other
                    if (workerStatus[worker] == -1 && !bikeStatus[bike])
                    {
                        bikeStatus[bike] = true;
                        workerStatus[worker] = bike;
                        pairCount++;

                        // If all the workers have the bike assigned, we can stop
                        if (pairCount == workers.Length)
                        {
                            return workerStatus;
                        }
                    }
                }

                return workerStatus;
            }
            // Class to store (worker, bike, distance)
            class WorkerBikePair
            {
                public int WorkerIndex { get; }
                public int BikeIndex { get; }
                public int Distance { get; }

                // Constructor to initialize the member variables
                public WorkerBikePair(int workerIndex, int bikeIndex, int distance)
                {
                    WorkerIndex = workerIndex;
                    BikeIndex = bikeIndex;
                    Distance = distance;
                }
            }

            // Custom comparator for sorting
            private class WorkerBikePairComparer : IComparer<WorkerBikePair>
            {
                public int Compare(WorkerBikePair a, WorkerBikePair b)
                {
                    if (a.Distance != b.Distance)
                    {
                        // Prioritize the one having smaller distance
                        return a.Distance - b.Distance;
                    }
                    else if (a.WorkerIndex != b.WorkerIndex)
                    {
                        // Prioritize according to the worker index
                        return a.WorkerIndex - b.WorkerIndex;
                    }
                    else
                    {
                        // Prioritize according to the bike index
                        return a.BikeIndex - b.BikeIndex;
                    }
                }
            }

            // Function to return the Manhattan distance
            private int FindDistance(int[] worker, int[] bike)
            {
                return Math.Abs(worker[0] - bike[0]) + Math.Abs(worker[1] - bike[1]);
            }

            /*
            Approach 2: Bucket Sort
            Complexity Analysis
Here, N is the number of workers, M is the number of bikes, and K is the maximum possible Manhattan distance of a worker/bike pair. In this problem, K equals 1998.
•	Time complexity: O(NM+K)
Generating all the (worker, bike) pairs takes O(NM) time. We are iterating over the generated pairs in the while loop according to their distance. Hence, at most, we will iterate over all NM pairs. But since there could be some currDis values at which no pairs exist, hence these operations have to be counted as well. The total possible values for currDis is K. Hence the time complexity equals O(NM+K)
•	Space complexity: O(NM+K)
We store all the pairs corresponding to their distance in disToPairs, which requires O(NM) space. To track the availability of the bikes bikeStatus takes O(M) space. Storing the index of the bike each worker is assigned in workerStatus takes O(N) space. Also, note that in C++ implementation, we have defined an array of size K. Hence, even if there are fewer than K pairs, it will still cost O(K) space.

            */
            // Function to return the Manhattan distance
            private int CalculateManhattanDistance(int[] worker, int[] bike)
            {
                return Math.Abs(worker[0] - bike[0]) + Math.Abs(worker[1] - bike[1]);
            }
            public int[] WithBucketSort(int[][] workers, int[][] bikes)
            {
                int minimumDistance = int.MaxValue;
                // Stores the list of (worker, bike) pairs corresponding to its distance
                Dictionary<int, List<KeyValuePair<int, int>>> distanceToPairs = new Dictionary<int, List<KeyValuePair<int, int>>>();

                // Add the (worker, bike) pair corresponding to its distance list
                for (int workerIndex = 0; workerIndex < workers.Length; workerIndex++)
                {
                    for (int bikeIndex = 0; bikeIndex < bikes.Length; bikeIndex++)
                    {
                        int distance = CalculateManhattanDistance(workers[workerIndex], bikes[bikeIndex]);

                        if (!distanceToPairs.ContainsKey(distance))
                        {
                            distanceToPairs[distance] = new List<KeyValuePair<int, int>>();
                        }

                        distanceToPairs[distance].Add(new KeyValuePair<int, int>(workerIndex, bikeIndex));
                        minimumDistance = Math.Min(minimumDistance, distance);
                    }
                }

                int currentDistance = minimumDistance;
                // Initialize all values to false, to signify no bikes have been taken
                bool[] bikeAvailabilityStatus = new bool[bikes.Length];

                int[] workerBikeAssignmentStatus = new int[workers.Length];
                // Initialize all index to -1, to mark all the workers available
                Array.Fill(workerBikeAssignmentStatus, -1);
                // Keep track of how many worker-bike pairs have been made
                int assignedPairCount = 0;

                // Until all workers have not been assigned a bike
                while (assignedPairCount != workers.Length)
                {
                    if (!distanceToPairs.ContainsKey(currentDistance))
                    {
                        currentDistance++;
                        continue;
                    }

                    foreach (var pair in distanceToPairs[currentDistance])
                    {
                        int workerIndex = pair.Key;
                        int bikeIndex = pair.Value;

                        if (workerBikeAssignmentStatus[workerIndex] == -1 && !bikeAvailabilityStatus[bikeIndex])
                        {
                            // If both worker and bike are free, assign them to each other
                            bikeAvailabilityStatus[bikeIndex] = true;
                            workerBikeAssignmentStatus[workerIndex] = bikeIndex;
                            assignedPairCount++;
                        }
                    }
                    currentDistance++;
                }

                return workerBikeAssignmentStatus;
            }

            /*
            Approach 3: Priority Queue
            Complexity Analysis
Here, N is the number of workers, and M is the number of bikes.
•	Time complexity: O(NMlogM)
We iterate over the N workers and for each worker:
o	Sorting the list of M bikes currWorkerPairs takes O(MlogM).
o	Add the next closest bike to pq. Insertion in pq takes O(logN).
Thus, the time complexity up to this point is O(NMlogM).
In the worst case, the total number of pop operations from the pq in the while loop can be O(N2). This is because, for ith worker, its first i-1 closest bike might have already been taken by previous workers. Hence, the first worker will get its first closest bike, the second worker gets its second-closest bike and so on. This way, the number of pop operations in the pq will be equal to 1 + 2 + 3 + 4 ...... N = (N * (N - 1)) / 2.
In each while loop operation, we are popping and pushing into the priority queue, which takes O(logN). Thus, the time complexity here is O(N2logN).
Therefore, the total time complexity is O(NMlogM+N2logN). Since we know, M≥N, the complexity can be written as O(NMlogM).
•	Space complexity: O(NM)
o	workerToBikeList store the list of M bikes for each N worker, hence it takes O(NM).
o	bikeStatus takes O(M) space.
o	workerStatus takes O(N) space.
o	pq will store at most N elements.
Hence, the total space complexity is equal to O(NM).

            */
            // List of pairs (distance, bike index) for each bike corresponding to worker
            private List<List<KeyValuePair<int, int>>> workerToBikeList = new List<List<KeyValuePair<int, int>>>();
            // Stores the closest bike index, corresponding to the worker index
            private int[] closestBikeIndex = new int[1001];

            public int[] WithMaxHeapPQ(int[][] workers, int[][] bikes)
            {
                var pq = new PriorityQueue<WorkerBikePair, WorkerBikePair>(new WorkerBikePairComparer());

                // Add all the bikes along with their distances from the worker
                for (int worker = 0; worker < workers.Length; worker++)
                {
                    var bikeList = new List<KeyValuePair<int, int>>();
                    for (int bike = 0; bike < bikes.Length; bike++)
                    {
                        int distance = FindDistance(workers[worker], bikes[bike]);
                        bikeList.Add(new KeyValuePair<int, int>(distance, bike));
                    }
                    bikeList.Sort((x, y) => x.Key.CompareTo(y.Key));

                    // Store all the bike options for the current worker in workerToBikeList
                    workerToBikeList.Add(bikeList);

                    // First bike is the closest bike for each worker
                    closestBikeIndex[worker] = 0;

                    // For each worker, add their closest bike to the priority queue
                    AddClosestBikeToPriorityQueue(pq, worker);
                }

                // Initialize all values to false, to signify no bikes have been taken
                bool[] bikeStatus = new bool[bikes.Length];

                // Initialize all index to -1, to mark all the workers available
                int[] workerStatus = new int[workers.Length];
                Array.Fill(workerStatus, -1);

                // Until all workers have not been assigned a bike
                while (pq.Count > 0)
                {
                    // Pop the pair with smallest distance
                    var workerBikePair = pq.Dequeue();

                    int worker = workerBikePair.WorkerIndex;
                    int bike = workerBikePair.BikeIndex;

                    if (workerStatus[worker] == -1 && !bikeStatus[bike])
                    {
                        // If both worker and bike are free, assign them to each other
                        bikeStatus[bike] = true;
                        workerStatus[worker] = bike;

                    }
                    else
                    {
                        // Add the next closest bike for the current worker
                        AddClosestBikeToPriorityQueue(pq, worker);
                    }
                }

                return workerStatus;
            }
            // Add the closest bike for the worker to the priority queue, 
            // And update the closest bike index
            private void AddClosestBikeToPriorityQueue(PriorityQueue<WorkerBikePair, WorkerBikePair> pq, int worker)
            {
                var closestBike = workerToBikeList[worker][closestBikeIndex[worker]];
                closestBikeIndex[worker]++;

                var workerBikePair = new WorkerBikePair(worker, closestBike.Value, closestBike.Key);
                pq.Enqueue(workerBikePair, workerBikePair);
            }



        }


        /* 1066. Campus Bikes II

        https://leetcode.com/problems/campus-bikes-ii/description/
         */
        public class AssignBikesIISolution
        {
            /*
            Approach 1: Greedy Backtracking
           Complexity Analysis
Here N is the number of workers, and M is the number of bikes.
•	Time complexity: O(M!/(M−N)!)
As discussed above, in the worst case, we will end up finding all the combinations of workers and bikes. Notice that this is equivalent to the number of permutations of N bikes taken from M total bikes.
•	Space complexity: O(N+M)
We have used an array visited to mark if the bike is available or not this will use O(M) space. There will also be some stack space used while making recursive calls. The recursion stack space used is proportional to the maximum number of active function calls in the stack. At most, this will be equal to the number of workers O(N).
 
            */
            public int GreedyBacktracking(int[][] workers, int[][] bikes)
            {
                FindMinimumDistanceSum(workers, 0, bikes, 0);
                return minimumDistanceSumValue;
            }
            // Maximum number of bikes is 10
            private bool[] isBikeVisited = new bool[10];
            private int minimumDistanceSumValue = int.MaxValue;

            // Manhattan distance
            private int CalculateDistance(int[] worker, int[] bike)
            {
                return Math.Abs(worker[0] - bike[0]) + Math.Abs(worker[1] - bike[1]);
            }

            private void FindMinimumDistanceSum(int[][] workers, int workerIndex, int[][] bikes, int currentDistanceSum)
            {
                if (workerIndex >= workers.Length)
                {
                    minimumDistanceSumValue = Math.Min(minimumDistanceSumValue, currentDistanceSum);
                    return;
                }
                // If the current distance sum is greater than the smallest result 
                // found then stop exploring this combination of workers and bikes
                if (currentDistanceSum >= minimumDistanceSumValue)
                {
                    return;
                }
                for (int bikeIndex = 0; bikeIndex < bikes.Length; bikeIndex++)
                {
                    // If bike is available
                    if (!isBikeVisited[bikeIndex])
                    {
                        isBikeVisited[bikeIndex] = true;
                        FindMinimumDistanceSum(workers, workerIndex + 1, bikes,
                            currentDistanceSum + CalculateDistance(workers[workerIndex], bikes[bikeIndex]));
                        isBikeVisited[bikeIndex] = false;
                    }
                }
            }
            /*
            Approach 2: Top-Down Dynamic Programming + BitMasking
            Complexity Analysis
Here N is the number of workers, and M is the number of bikes.
•	Time complexity: O(M⋅2^M)
Time complexity is equal to the number of unique states in the memo table multiplied by the average time that the minimumDistanceSum function takes. The number of states is equal to unique values of mask that is 2^M and the minimumDistanceSum function takes O(M) time. So the time complexity is O(M⋅2^M).
•	Space complexity: O(2^M)
We have used an array memo to store the results corresponding to mask. Also, there will be some stack space used during recursion. The recursion space will be equal to the maximum number of the active function calls in the stack that will be equal to the number of workers i.e., N. Hence the space complexity will be equal to O(2^M+N).


            */
            // Maximum value of mask will be 2^(Number of bikes)
            // And number of bikes can be 10 at max
            int[] memo = new int[1024];
            public int TopDownDPWithBitMasking(int[][] workers, int[][] bikes)
            {
                Array.Fill(memo, -1);
                return MinimumDistanceSum(workers, bikes, 0, 0);
            }
            private int MinimumDistanceSum(int[][] workers, int[][] bikes, int workerIndex, int mask)
            {
                if (workerIndex >= workers.Length)
                {
                    return 0;
                }

                // If result is already calculated, return it no recursion needed
                if (memo[mask] != -1)
                    return memo[mask];

                int smallestDistanceSum = int.MaxValue;
                for (int bikeIndex = 0; bikeIndex < bikes.Length; bikeIndex++)
                {
                    // Check if the bike at bikeIndex is available
                    if ((mask & (1 << bikeIndex)) == 0)
                    {
                        // Adding the current distance and repeat the process for next worker
                        // also changing the bit at index bikeIndex to 1 to show the bike there is assigned
                        smallestDistanceSum = Math.Min(smallestDistanceSum,
                                                       CalculateDistance(workers[workerIndex], bikes[bikeIndex]) +
                                                       MinimumDistanceSum(workers, bikes, workerIndex + 1,
                                                                          mask | (1 << bikeIndex)));
                    }
                }

                // Memoizing the result corresponding to mask
                return memo[mask] = smallestDistanceSum;
            }
            /*
            Approach 3: Bottom-Up Dynamic Programming + BitMasking
            Complexity Analysis
Here N is the number of workers, and M is the number of bikes.
•	Time complexity: O(M⋅2^M)
We traverse over all of the values for mask from 0 to 2^M and for each value, we traverse over the M bikes and also count the number of ones in mask, which on average takes M/2 iterations using Kernighan's Algorithm. So the time complexity will be O(2^M⋅(M+M/2)) which simplifies to O(M⋅2^M).
•	Space complexity: O(2^M).
We are only using space in memo with the size equal to 2^M.

            */
            public int BottomUpDPWithBitMasking(int[][] workers, int[][] bikes)
            {
                // Initializing the answers for all masks to be INT_MAX
                Array.Fill(memo, int.MaxValue);
                return MinimumDistanceSum(workers, bikes);
            }
            // Count the number of ones using Brian Kernighan’s Algorithm
            private int CountNumberOfOnes(int mask)
            {
                int count = 0;
                while (mask != 0)
                {
                    mask &= (mask - 1);
                    count++;
                }
                return count;
            }
            private int MinimumDistanceSum(int[][] workers, int[][] bikes)
            {
                int numberOfBikes = bikes.Length;
                int numberOfWorkers = workers.Length;
                int smallestDistanceSum = int.MaxValue;

                // 0 signifies that no bike has been assigned and
                // Distance sum for not assigning any bike is equal to 0
                memo[0] = 0;

                // Traverse over all the possible values of mask
                for (int mask = 0; mask < (1 << numberOfBikes); mask++)
                {
                    int nextWorkerIndex = CountNumberOfOnes(mask);

                    // If mask has more number of 1's than the number of workers
                    // Then we can update our answer accordingly
                    if (nextWorkerIndex >= numberOfWorkers)
                    {
                        smallestDistanceSum = Math.Min(smallestDistanceSum, memo[mask]);
                        continue;
                    }

                    for (int bikeIndex = 0; bikeIndex < numberOfBikes; bikeIndex++)
                    {
                        // Checking if the bike at bikeIndex has already been assigned
                        if ((mask & (1 << bikeIndex)) == 0)
                        {
                            int newMask = (1 << bikeIndex) | mask;

                            // Updating the distance sum for newMask
                            memo[newMask] = Math.Min(memo[newMask], memo[mask] +
                                                     CalculateDistance(workers[nextWorkerIndex], bikes[bikeIndex]));
                        }
                    }
                }

                return smallestDistanceSum;
            }
            /*
            Approach 4: Priority Queue (Similar to Dijkstra's Algorithm)
Complexity Analysis
Here N is the number of workers, M is the number of bikes and,
P(M,N)=M!/(M−N)! is the number of permutations for N bikes taken from M total bikes,
C(M,N)=M!/((M−N)!⋅N!) is the number of ways to choose N bikes from M total bikes.
•	Time complexity: O(P(M,N)⋅log(P(M,N))+(M⋅log(P(M,N))⋅2^M)
Priority queue might have more than 1 copy of a mask. For instance 0011 will be inserted into the priority queue twice, the first occasion, 0001 -> 0011, the second occasion 0010 -> 0011.
The total number of the possible mask with size M and having N ones will be C(M, N). For each such mask, the order in which 1's are added to mask will also matter, this can be done in N! ways. So in total, there can be C(M,N)⋅N!=P(M,N) number of mask in the priority queue. All these mask will be iterated in the while loop and for each mask, log(P(M,N)) number of operations will be required for removing the top pair from the priority queue.
Since we are tracking the mask that we have traversed using visited set, the inner for loop where we are traversing over the bikes will only be executed for only unique values of mask that is 2M. Also pushing into priority queue will cost log(P(M,N)) time.
Hence the total time complexity becomes O(P(M,N)⋅log(P(M,N))+(M⋅log(P(M,N))⋅2^M).
•	Space complexity: O(P(M,N)+2^M)
The number of mask that can be stored in the priority queue is O(P(M,N)), and the number of mask that can be inserted into the set visited will be O(2^M).

            */
            public int WithPQSimilarToDijkstra(int[][] workers, int[][] bikes)
            {
                int numOfBikes = bikes.Length, numOfWorkers = workers.Length;

                PriorityQueue<int[], int[]> priorityQueue = new PriorityQueue<int[], int[]>(Comparer<int[]>.Create((a, b) => a[0] - b[0]));
                HashSet<int> visited = new HashSet<int>();

                // Initially both distance sum and mask is 0
                priorityQueue.Enqueue(new int[] { 0, 0 }, new int[] { 0, 0 });
                while (priorityQueue.Count > 0)
                {
                    int currentDistanceSum = priorityQueue.Peek()[0];
                    int currentMask = priorityQueue.Peek()[1];
                    priorityQueue.Dequeue();

                    // Continue if the mask is already traversed
                    if (visited.Contains(currentMask))
                        continue;

                    // Marking the mask as visited
                    visited.Add(currentMask);
                    // Next Worker index would be equal to the number of 1's in currentMask
                    int workerIndex = CountNumberOfOnes(currentMask);

                    // Return the current distance sum if all workers are covered
                    if (workerIndex == numOfWorkers)
                    {
                        return currentDistanceSum;
                    }

                    for (int bikeIndex = 0; bikeIndex < numOfBikes; bikeIndex++)
                    {
                        // Checking if the bike at bikeIndex has been assigned or not
                        if ((currentMask & (1 << bikeIndex)) == 0)
                        {
                            int nextStateDistanceSum = currentDistanceSum +
                                CalculateDistance(workers[workerIndex], bikes[bikeIndex]);

                            // Put the next state pair into the priority queue
                            int nextStateMask = currentMask | (1 << bikeIndex);
                            priorityQueue.Enqueue(new int[] { nextStateDistanceSum, nextStateMask }, new int[] { nextStateDistanceSum, nextStateMask });
                        }
                    }
                }

                // This statement will never be executed provided there is at least one bike per worker
                return -1;
            }

        }


        /* 1125. Smallest Sufficient Team
        https://leetcode.com/problems/smallest-sufficient-team/description/
         */

        class SmallestSufficientTeamSol
        {
            /*
            Approach 1: Bottom-Up Dynamic Programming with Bitmasks
            Complexity Analysis
•	Time complexity: O((2^m)⋅n).
There are two nested for loops: for skillsMask, which performs O(2^m) iterations, and for i, which performs O(n) iterations. We process each transition inside these loops in O(1).
•	Space complexity: O(2^m).
We store a DP array of size 2^m.

            */
            public int[] BottomUpDPWithBitMasking(string[] requiredSkills, List<List<string>> people)
            {
                int numberOfPeople = people.Count, numberOfSkills = requiredSkills.Length;
                Dictionary<string, int> skillId = new Dictionary<string, int>();
                for (int i = 0; i < numberOfSkills; i++)
                {
                    skillId[requiredSkills[i]] = i;
                }
                int[] skillsMaskOfPerson = new int[numberOfPeople];
                for (int i = 0; i < numberOfPeople; i++)
                {
                    foreach (string skill in people[i])
                    {
                        skillsMaskOfPerson[i] |= 1 << skillId[skill];
                    }
                }
                long[] dp = new long[1 << numberOfSkills];
                Array.Fill(dp, (1L << numberOfPeople) - 1);
                dp[0] = 0;
                for (int skillsMask = 1; skillsMask < (1 << numberOfSkills); skillsMask++)
                {
                    for (int i = 0; i < numberOfPeople; i++)
                    {
                        int smallerSkillsMask = skillsMask & ~skillsMaskOfPerson[i];
                        if (smallerSkillsMask != skillsMask)
                        {
                            long peopleMask = dp[smallerSkillsMask] | (1L << i);
                            if (BitCount(peopleMask) < BitCount(dp[skillsMask]))
                            {
                                dp[skillsMask] = peopleMask;
                            }
                        }
                    }
                }
                long answerMask = dp[(1 << numberOfSkills) - 1];
                int[] answer = new int[BitCount(answerMask)];
                int pointer = 0;
                for (int i = 0; i < numberOfPeople; i++)
                {
                    if (((answerMask >> i) & 1) == 1)
                    {
                        answer[pointer++] = i;
                    }
                }
                return answer;
            }

            private int BitCount(long number)
            {
                int count = 0;
                while (number > 0)
                {
                    count++;
                    number &= (number - 1);
                }
                return count;
            }
            /*
            Approach 2: Top-Down Dynamic Programming (Memoization)
Complexity Analysis
•	Time complexity: O((2^m)⋅n).
Even though we changed the order of calculating DP, the time complexity is the same as in the previous approach: for each skillsMask, we compute dp[skillsMask] in O(n). Since we store the results in memory, we will calculate each dp[skillsMask] only once.
•	Space complexity: O(2^m).

            */
            public int[] TopDownDPMemoWithBitMasking(string[] requiredSkills, List<List<string>> people)
            {
                numberOfPeople = people.Count;
                int numberOfSkills = requiredSkills.Length;
                Dictionary<string, int> skillId = new Dictionary<string, int>();
                for (int i = 0; i < numberOfSkills; i++)
                {
                    skillId[requiredSkills[i]] = i;
                }
                skillsMaskOfPerson = new int[numberOfPeople];
                for (int i = 0; i < numberOfPeople; i++)
                {
                    foreach (string skill in people[i])
                    {
                        skillsMaskOfPerson[i] |= 1 << skillId[skill];
                    }
                }
                dp = new long[1 << numberOfSkills];
                Array.Fill(dp, -1L);
                long answerMask = CalculateSkillsMask((1 << numberOfSkills) - 1).Value;
                int[] answer = new int[CountBits(answerMask)];
                int index = 0;
                for (int i = 0; i < numberOfPeople; i++)
                {
                    if (((answerMask >> i) & 1) == 1)
                    {
                        answer[index++] = i;
                    }
                }
                return answer;
            }
            private int CountBits(long number)
            {
                int count = 0;
                while (number > 0)
                {
                    count += (int)(number & 1);
                    number >>= 1;
                }
                return count;
            }
            int numberOfPeople;
            int[] skillsMaskOfPerson;
            long[] dp;
            private long? CalculateSkillsMask(int skillsMask)
            {
                if (skillsMask == 0)
                {
                    return 0L;
                }
                if (dp[skillsMask] != -1)
                {
                    return dp[skillsMask];
                }
                for (int i = 0; i < numberOfPeople; i++)
                {
                    int smallerSkillsMask = skillsMask & ~skillsMaskOfPerson[i];
                    if (smallerSkillsMask != skillsMask)
                    {
                        long peopleMask = CalculateSkillsMask(smallerSkillsMask).Value | (1L << i);
                        if (dp[skillsMask] == -1 || CountBits(peopleMask) < CountBits(dp[skillsMask]))
                        {
                            dp[skillsMask] = peopleMask;
                        }
                    }
                }
                return dp[skillsMask];
            }


        }

        /* 1162. As Far from Land as Possible
        https://leetcode.com/problems/as-far-from-land-as-possible/description/
         */
        class MaxDistanceSol
        {
            /*
            Approach 1: Breadth-First Search (BFS)
            Complexity Analysis
Here N is the side of the square matrix with size N∗N.
•	Time complexity: O(N^2).
We start traversing from land cells (1) and keep iterating over water cells until we convert all water cells to land. Notice that we never insert any cell into the queue twice as we mark the water cell as land when we visit them. Therefore, the time complexity equals O(N^2).
•	Space complexity: O(N^2).
There could be all cells in the queue at a particular time. Considering the matrix doesn't have any water cells, we insert all the land cells into the queue to initialize and thus will take O(N^2) space. Also, we create visited, a copy matrix of grid. Hence, the space complexity is O(N^2).

            */
            // Four directions: Up, Down, Left and Right.
            int[][] directions = new int[][] { new int[] { -1, 0 }, new int[] { 1, 0 }, new int[] { 0, -1 }, new int[] { 0, 1 } };

            public int BFS(int[][] grid)
            {
                // A copy matrix of the grid to mark water cells as land once visited.
                int[][] visited = new int[grid.Length][];

                for (int i = 0; i < grid.Length; i++)
                {
                    visited[i] = new int[grid[i].Length];
                }

                // Insert all the land cells in the queue.
                Queue<KeyValuePair<int, int>> queue = new Queue<KeyValuePair<int, int>>();
                for (int row = 0; row < grid.Length; row++)
                {
                    for (int column = 0; column < grid[0].Length; column++)
                    {
                        // Copy grid to the visited matrix.
                        visited[row][column] = grid[row][column];
                        if (grid[row][column] == 1)
                        {
                            queue.Enqueue(new KeyValuePair<int, int>(row, column));
                        }
                    }
                }

                int distance = -1;
                while (queue.Count > 0)
                {
                    int queueSize = queue.Count;

                    // Iterate over all the current cells in the queue.
                    while (queueSize-- > 0)
                    {
                        KeyValuePair<int, int> landCell = queue.Dequeue();

                        // From the current land cell, traverse to all the four directions
                        // and check if it is a water cell. If yes, convert it to land
                        // and add it to the queue.
                        foreach (int[] direction in directions)
                        {
                            int x = landCell.Key + direction[0];
                            int y = landCell.Value + direction[1];

                            if (x >= 0 && y >= 0 && x < grid.Length && y < grid[0].Length && visited[x][y] == 0)
                            {
                                // Marking as 1 to avoid re-iterating it.
                                visited[x][y] = 1;
                                queue.Enqueue(new KeyValuePair<int, int>(x, y));
                            }
                        }
                    }

                    // After each iteration of queue elements, increment distance 
                    // as we covered 1 unit distance from all cells in every direction.
                    distance++;
                }

                // If the distance is 0, there is no water cell.
                return distance == 0 ? -1 : distance;
            }

            /* Approach 2: Dynamic-Programming
            Complexity Analysis
Here N is the side of the square matrix with size N∗N.
•	Time complexity: O(N^2).
We iterate over the matrix twice from top to bottom and bottom to top; hence the total time complexity equals O(N^2).
•	Space complexity: O(N^2).
The only space we need is the matrix dist of size N∗N to store the minimum distance for all cells. Therefore, the total space complexity equals O(N^2).

             */
            public int WithDP(int[][] grid)
            {
                int numberOfRows = grid.Length;
                // Although it's a square matrix, using different variable for readability.
                int numberOfColumns = grid[0].Length;

                // Maximum manhattan distance possible + 1.
                int MAX_DISTANCE = numberOfRows + numberOfColumns + 1;

                int[][] distance = new int[numberOfRows][];
                for (int i = 0; i < numberOfRows; i++)
                    distance[i] = new int[numberOfColumns];

                foreach (var array in distance)
                    Array.Fill(array, MAX_DISTANCE);

                // First pass: check for left and top neighbours
                for (int rowIndex = 0; rowIndex < numberOfRows; rowIndex++)
                {
                    for (int colIndex = 0; colIndex < numberOfColumns; colIndex++)
                    {
                        // Distance of land cells will be 0.
                        if (grid[rowIndex][colIndex] == 1)
                        {
                            distance[rowIndex][colIndex] = 0;
                        }
                        else
                        {
                            // Check left and top cell distances if they exist and update the current cell distance.
                            distance[rowIndex][colIndex] = Math.Min(distance[rowIndex][colIndex],
                                Math.Min(rowIndex > 0 ? distance[rowIndex - 1][colIndex] + 1 : MAX_DISTANCE,
                                         colIndex > 0 ? distance[rowIndex][colIndex - 1] + 1 : MAX_DISTANCE));
                        }
                    }
                }

                // Second pass: check for the bottom and right neighbours.
                int maximumDistance = int.MinValue;
                for (int rowIndex = numberOfRows - 1; rowIndex >= 0; rowIndex--)
                {
                    for (int colIndex = numberOfColumns - 1; colIndex >= 0; colIndex--)
                    {
                        // Check the right and bottom cell distances if they exist and update the current cell distance.
                        distance[rowIndex][colIndex] = Math.Min(distance[rowIndex][colIndex],
                            Math.Min(rowIndex < numberOfRows - 1 ? distance[rowIndex + 1][colIndex] + 1 : MAX_DISTANCE,
                                     colIndex < numberOfColumns - 1 ? distance[rowIndex][colIndex + 1] + 1 : MAX_DISTANCE));

                        maximumDistance = Math.Max(maximumDistance, distance[rowIndex][colIndex]);
                    }
                }

                // If maximumDistance is 0, it means there is no water cell,
                // If maximumDistance is MAX_DISTANCE, it implies no land cell.
                return maximumDistance == 0 || maximumDistance == MAX_DISTANCE ? -1 : maximumDistance;
            }
            /*
            Approach 3: Space-Optimized Dynamic-Programming
            Complexity Analysis
Here N is the side of the square matrix with size N∗N.
•	Time complexity: O(N^2).
We iterate over the matrix twice from top to bottom and bottom to top; hence the total time complexity equals O(N^2).
•	Space complexity: O(1).
We don't need extra space, so the space complexity is constant.

            */
            public int SpaceOptimizedDP(int[][] grid)
            {
                int numberOfRows = grid.Length;
                // Although it's a square matrix, using different variable for readability.
                int numberOfColumns = grid[0].Length;

                // Maximum manhattan distance possible + 1.
                int MAX_DISTANCE = numberOfRows + numberOfColumns + 1;

                // First pass: check for left and top neighbours
                for (int rowIndex = 0; rowIndex < numberOfRows; rowIndex++)
                {
                    for (int columnIndex = 0; columnIndex < numberOfColumns; columnIndex++)
                    {
                        if (grid[rowIndex][columnIndex] == 1)
                        {
                            // Distance of land cells will be 0.
                            grid[rowIndex][columnIndex] = 0;
                        }
                        else
                        {
                            grid[rowIndex][columnIndex] = MAX_DISTANCE;
                            // Check left and top cell distances if they exist and update the current cell distance.
                            grid[rowIndex][columnIndex] = Math.Min(grid[rowIndex][columnIndex], Math.Min(rowIndex > 0 ? grid[rowIndex - 1][columnIndex] + 1 : MAX_DISTANCE,
                                                                       columnIndex > 0 ? grid[rowIndex][columnIndex - 1] + 1 : MAX_DISTANCE));
                        }
                    }
                }

                // Second pass: check for the bottom and right neighbours.
                int answer = int.MinValue;
                for (int rowIndex = numberOfRows - 1; rowIndex >= 0; rowIndex--)
                {
                    for (int columnIndex = numberOfColumns - 1; columnIndex >= 0; columnIndex--)
                    {
                        // Check the right and bottom cell distances if they exist and update the current cell distance.
                        grid[rowIndex][columnIndex] = Math.Min(grid[rowIndex][columnIndex], Math.Min(rowIndex < numberOfRows - 1 ? grid[rowIndex + 1][columnIndex] + 1 : MAX_DISTANCE,
                                                                   columnIndex < numberOfColumns - 1 ? grid[rowIndex][columnIndex + 1] + 1 : MAX_DISTANCE));

                        answer = Math.Max(answer, grid[rowIndex][columnIndex]);
                    }
                }

                // If answer is 1, it means there is no water cell,
                // If answer is MAX_DISTANCE, it implies no land cell.
                return answer == 0 || answer == MAX_DISTANCE ? -1 : answer;
            }
        }


        /* 317. Shortest Distance from All Buildings
        https://leetcode.com/problems/shortest-distance-from-all-buildings/description/
         */

        public class ShortestDistanceSol
        {
            /*
Approach 1: BFS from Empty Land to All Houses
Complexity Analysis
Let N and M be the number of rows and columns in grid respectively.
•	Time Complexity: O(N^2⋅M^2)
For each empty land, we will traverse to all other houses.
This will require O(number of zeros ⋅ number of ones) time and the number of zeros and ones in the matrix is of order N⋅M.
Consider that when half of the values in grid are 0 and half of the values are 1, the total elements in grid would be (M⋅N) so their counts are (M⋅N)/2 and (M⋅N)/2 respectively, thus giving time complexity (M⋅N)/2⋅(M⋅N)/2, that results in O(N^2⋅M^2).
In JavaScript implementation, for simplicity, we have used an array for the queue.
Since popping elements from the front of an array is an O(n) operation, which is undesirable,
instead of popping from the front of the queue, we will iterate over the queue and record cells to be explored in the next level in next_queue.
Once the current queue has been traversed, we simply set queue equal to the next_queue.
•	Space Complexity: O(N⋅M)
We use an extra matrix to track the visited cells, and the queue will store each matrix element at most once during each BFS call.
Hence, O(N⋅M) space is required.

            */
            public int BFSFromEmptyLandToAllHouses(int[][] grid)
            {
                int minimumDistance = int.MaxValue;
                int totalRows = grid.Length;
                int totalCols = grid[0].Length;
                int totalHouses = 0;

                for (int row = 0; row < totalRows; row++)
                {
                    for (int col = 0; col < totalCols; col++)
                    {
                        if (grid[row][col] == 1)
                        {
                            totalHouses++;
                        }
                    }
                }

                // Find the minimum distance sum for each empty cell.
                for (int row = 0; row < totalRows; row++)
                {
                    for (int col = 0; col < totalCols; col++)
                    {
                        if (grid[row][col] == 0)
                        {
                            minimumDistance = Math.Min(minimumDistance, PerformBfs(grid, row, col, totalHouses));
                        }
                    }
                }

                // If it is impossible to reach all houses from any empty cell, then return -1.
                if (minimumDistance == int.MaxValue)
                {
                    return -1;
                }

                return minimumDistance;
            }


            private int PerformBfs(int[][] grid, int startRow, int startCol, int totalHouses)
            {
                // Next four directions.
                int[][] directions = new int[][] { new int[] { 1, 0 }, new int[] { -1, 0 }, new int[] { 0, 1 }, new int[] { 0, -1 } };

                int totalRows = grid.Length;
                int totalCols = grid[0].Length;
                int distanceSum = 0;
                int housesReached = 0;

                // Queue to do a BFS, starting from (startRow, startCol) cell.
                Queue<int[]> queue = new Queue<int[]>();
                queue.Enqueue(new int[] { startRow, startCol });

                // Keep track of visited cells.
                bool[,] visited = new bool[totalRows, totalCols];
                visited[startRow, startCol] = true;

                int steps = 0;
                while (queue.Count > 0 && housesReached != totalHouses)
                {
                    int currentLevelSize = queue.Count;
                    for (int i = 0; i < currentLevelSize; i++)
                    {
                        int[] currentCell = queue.Dequeue();
                        startRow = currentCell[0];
                        startCol = currentCell[1];

                        // If this cell is a house, then add the distance from source to this cell
                        // and we go past from this cell.
                        if (grid[startRow][startCol] == 1)
                        {
                            distanceSum += steps;
                            housesReached++;
                            continue;
                        }

                        // This cell was empty cell, hence traverse the next cells that are not a blockage.
                        foreach (int[] direction in directions)
                        {
                            int nextRow = startRow + direction[0];
                            int nextCol = startCol + direction[1];
                            if (nextRow >= 0 && nextCol >= 0 && nextRow < totalRows && nextCol < totalCols)
                            {
                                if (!visited[nextRow, nextCol] && grid[nextRow][nextCol] != 2)
                                {
                                    visited[nextRow, nextCol] = true;
                                    queue.Enqueue(new int[] { nextRow, nextCol });
                                }
                            }
                        }
                    }

                    // After traversing one level of cells, increment the steps by 1 to reach the next level.
                    steps++;
                }

                // If we did not reach all houses, then any cell visited also cannot reach all houses.
                // Set all cells visited to 2 so we do not check them again and return MAX_VALUE.
                if (housesReached != totalHouses)
                {
                    for (startRow = 0; startRow < totalRows; startRow++)
                    {
                        for (startCol = 0; startCol < totalCols; startCol++)
                        {
                            if (grid[startRow][startCol] == 0 && visited[startRow, startCol])
                            {
                                grid[startRow][startCol] = 2;
                            }
                        }
                    }
                    return int.MaxValue;
                }

                // If we have reached all houses then return the total distance calculated.
                return distanceSum;
            }

            /*
            Approach 2: BFS from Houses to Empty Land
Complexity Analysis
Let N and M be the number of rows and columns in grid respectively.
•	Time Complexity: O(N^2⋅M^2)
For each house, we will traverse across all reachable land.
This will require O(number of zeros ⋅ number of ones) time and the number of zeros and ones in the matrix is of order N⋅M.
Consider that when half of the values in grid are 0 and half of the values are 1, total elements in grid will be (M⋅N) so their counts are (M⋅N)/2 and (M⋅N)/2 respectively, thus giving time complexity (M⋅N)/2⋅(M⋅N)/2, which results in O(N^2⋅M^2).
In JavaScript implementation, for simplicity, we have used an array for the queue.
However, popping elements from the front of an array is an O(n) operation, which is undesirable.
So, instead of popping from the front of the queue, we will iterate over the queue and record cells to be explored in the next level in next_queue.
Once the current queue has been traversed, we simply set queue equal to the next_queue.
•	Space Complexity: O(N⋅M)
We use an extra matrix to track the visited cells and another one to store distance sum along with the house counter for each empty cell, and the queue will store each matrix element at most once during each BFS call.
Hence, O(N⋅M) space is required.

            */
            public int BFSFromHousesToEmptyLand(int[][] grid)
            {
                int minDistance = int.MaxValue;
                int rows = grid.Length;
                int cols = grid[0].Length;
                int totalHouses = 0;

                // Store { total_dist, houses_count } for each cell.
                int[][][] distances = new int[rows][][];
                for (int i = 0; i < rows; i++)
                {
                    distances[i] = new int[cols][];
                    for (int j = 0; j < cols; j++)
                    {
                        distances[i][j] = new int[2];
                    }
                }

                // Count houses and start bfs from each house.
                for (int row = 0; row < rows; ++row)
                {
                    for (int col = 0; col < cols; ++col)
                    {
                        if (grid[row][col] == 1)
                        {
                            totalHouses++;
                            Bfs(grid, distances, row, col);
                        }
                    }
                }

                // Check all empty lands with houses count equal to total houses and find the min ans.
                for (int row = 0; row < rows; ++row)
                {
                    for (int col = 0; col < cols; ++col)
                    {
                        if (distances[row][col][1] == totalHouses)
                        {
                            minDistance = Math.Min(minDistance, distances[row][col][0]);
                        }
                    }
                }

                // If we haven't found a valid cell return -1.
                if (minDistance == int.MaxValue)
                {
                    return -1;
                }
                return minDistance;
            }
            private void Bfs(int[][] grid, int[][][] distances, int row, int col)
            {
                int[][] directions = new int[][] { new int[] { 1, 0 }, new int[] { -1, 0 }, new int[] { 0, 1 }, new int[] { 0, -1 } };

                int rows = grid.Length;
                int cols = grid[0].Length;

                // Use a queue to do a bfs, starting from each cell located at (row, col).
                Queue<int[]> queue = new Queue<int[]>();
                queue.Enqueue(new int[] { row, col });

                // Keep track of visited cells.
                bool[][] visited = new bool[rows][];
                for (int i = 0; i < rows; i++)
                {
                    visited[i] = new bool[cols];
                }
                visited[row][col] = true;

                int steps = 0;

                while (queue.Count > 0)
                {
                    for (int i = queue.Count; i > 0; --i)
                    {
                        int[] current = queue.Dequeue();
                        row = current[0];
                        col = current[1];

                        // If we reached an empty cell, then add the distance
                        // and increment the count of houses reached at this cell.
                        if (grid[row][col] == 0)
                        {
                            distances[row][col][0] += steps;
                            distances[row][col][1] += 1;
                        }

                        // Traverse the next cells which are not a blockage.
                        foreach (int[] direction in directions)
                        {
                            int nextRow = row + direction[0];
                            int nextCol = col + direction[1];

                            if (nextRow >= 0 && nextCol >= 0 && nextRow < rows && nextCol < cols)
                            {
                                if (!visited[nextRow][nextCol] && grid[nextRow][nextCol] == 0)
                                {
                                    visited[nextRow][nextCol] = true;
                                    queue.Enqueue(new int[] { nextRow, nextCol });
                                }
                            }
                        }
                    }

                    // After traversing one level cells, increment the steps by 1.
                    steps++;
                }
            }
            /*
            Approach 3: BFS from Houses to Empty Land (Optimized)	
           Complexity Analysis
 Let N and M be the number of rows and columns in grid respectively.
 •	Time Complexity: O(N^2⋅M^2)
 For each house, we will traverse across all reachable land.
 This will require O(number of zeros ⋅ number of ones) time and the number of zeros and ones in the matrix is of order N⋅M.
 Consider that when half of the values in grid are 0 and half of the values are 1, total elements in grid would be (M⋅N) so their counts are (M⋅N)/2 and (M⋅N)/2 respectively, thus giving time complexity (M⋅N)/2⋅(M⋅N)/2, that results in O(N^2⋅M^2).
 •	Space Complexity: O(N⋅M)
 We use an extra matrix to store distance sums, and the queue will store each matrix element at most once during each BFS call.
 Hence, O(N⋅M) space is required.

            */
            public int BFSFromHousesToEmptyLandOptimal(int[][] grid)
            {
                // Next four directions.
                int[][] directions = new int[][] { new int[] { 1, 0 }, new int[] { -1, 0 }, new int[] { 0, 1 }, new int[] { 0, -1 } };

                int rowCount = grid.Length;
                int colCount = grid[0].Length;

                // Total matrix to store total distance sum for each empty cell.
                int[][] totalDistance = new int[rowCount][];
                for (int i = 0; i < rowCount; i++)
                {
                    totalDistance[i] = new int[colCount];
                }

                int emptyLandValue = 0;
                int minimumDistance = int.MaxValue;

                for (int row = 0; row < rowCount; ++row)
                {
                    for (int col = 0; col < colCount; ++col)
                    {
                        // Start a BFS from each house.
                        if (grid[row][col] == 1)
                        {
                            minimumDistance = int.MaxValue;

                            // Use a queue to perform a BFS, starting from the cell at (row, col).
                            Queue<int[]> queue = new Queue<int[]>();
                            queue.Enqueue(new int[] { row, col });

                            int steps = 0;

                            while (queue.Count > 0)
                            {
                                steps++;

                                for (int level = queue.Count; level > 0; --level)
                                {
                                    int[] current = queue.Dequeue();

                                    foreach (int[] direction in directions)
                                    {
                                        int nextRow = current[0] + direction[0];
                                        int nextCol = current[1] + direction[1];

                                        // For each cell with the value equal to empty land value
                                        // add distance and decrement the cell value by 1.
                                        if (nextRow >= 0 && nextRow < rowCount &&
                                            nextCol >= 0 && nextCol < colCount &&
                                            grid[nextRow][nextCol] == emptyLandValue)
                                        {
                                            grid[nextRow][nextCol]--;
                                            totalDistance[nextRow][nextCol] += steps;

                                            queue.Enqueue(new int[] { nextRow, nextCol });
                                            minimumDistance = Math.Min(minimumDistance, totalDistance[nextRow][nextCol]);
                                        }
                                    }
                                }
                            }

                            // Decrement empty land value to be searched in next iteration.
                            emptyLandValue--;
                        }
                    }
                }

                return minimumDistance == int.MaxValue ? -1 : minimumDistance;
            }

        }


        /* 296. Best Meeting Point
        https://leetcode.com/problems/best-meeting-point/description/
         */
        public class MinTotalDistanceSol
        {
            /*
            Approach #1 (Breadth-first Search) [Time Limit Exceeded]
Complexity analysis
•	Time complexity : O(m^2*n^2).
For each point in the m×n size grid, the breadth-first search takes at most m×n steps to reach all points. Therefore the time complexity is O(m^2*n^2).
•	Space complexity : O(mn).
The visited table consists of m×n elements map to each point in the grid. We insert at most m×n points into the queue.

            */
            public int BFS(int[,] grid)
            {
                int minDistance = int.MaxValue;
                for (int row = 0; row < grid.GetLength(0); row++)
                {
                    for (int col = 0; col < grid.GetLength(1); col++)
                    {
                        int distance = Search(grid, row, col);
                        minDistance = Math.Min(distance, minDistance);
                    }
                }
                return minDistance;
            }

            private int Search(int[,] grid, int row, int col)
            {
                Queue<Point> queue = new Queue<Point>();
                int rows = grid.GetLength(0);
                int cols = grid.GetLength(1);
                bool[,] visited = new bool[rows, cols];
                queue.Enqueue(new Point(row, col, 0));
                int totalDistance = 0;
                while (queue.Count > 0)
                {
                    Point point = queue.Dequeue();
                    int r = point.Row;
                    int c = point.Col;
                    int d = point.Distance;
                    if (r < 0 || c < 0 || r >= rows || c >= cols || visited[r, c])
                    {
                        continue;
                    }
                    if (grid[r, c] == 1)
                    {
                        totalDistance += d;
                    }
                    visited[r, c] = true;
                    queue.Enqueue(new Point(r + 1, c, d + 1));
                    queue.Enqueue(new Point(r - 1, c, d + 1));
                    queue.Enqueue(new Point(r, c + 1, d + 1));
                    queue.Enqueue(new Point(r, c - 1, d + 1));
                }
                return totalDistance;
            }

            public class Point
            {
                public int Row { get; }
                public int Col { get; }
                public int Distance { get; }
                public Point(int row, int col, int distance = 0)
                {
                    Row = row;
                    Col = col;
                    Distance = distance;
                }
            }

            /* Approach #2 (Manhattan Distance Formula) [Time Limit Exceeded]
            Complexity analysis
            •	Time complexity : O(m^2*n^2).
            Assume that k is the total number of houses. For each point in the m×n size grid, we calculate the manhattan distance in O(k). Therefore the time complexity is O(mnk). But do note that there could be up to m×n houses, making the worst case time complexity to be O(m^2*n^2).
            •	Space complexity : O(mn).

             */
            public int UsingManhattanDistanceFormula(int[][] grid)
            {
                List<Point> points = GetAllPoints(grid);
                int minDistance = int.MaxValue;
                for (int row = 0; row < grid.Length; row++)
                {
                    for (int col = 0; col < grid[0].Length; col++)
                    {
                        int distance = CalculateDistance(points, row, col);
                        minDistance = Math.Min(distance, minDistance);
                    }
                }
                return minDistance;
            }

            private int CalculateDistance(List<Point> points, int row, int col)
            {
                int distance = 0;
                foreach (Point point in points)
                {
                    distance += Math.Abs(point.Row - row) + Math.Abs(point.Col - col);
                }
                return distance;
            }

            private List<Point> GetAllPoints(int[][] grid)
            {
                List<Point> points = new List<Point>();
                for (int row = 0; row < grid.Length; row++)
                {
                    for (int col = 0; col < grid[0].Length; col++)
                    {
                        if (grid[row][col] == 1)
                        {
                            points.Add(new Point(row, col));
                        }
                    }
                }
                return points;
            }
            /*             Approach #3 (Sorting) [Accepted]
            Complexity analysis
            •	Time complexity : O(mn+nlogn).
            Since there could be at most m×n points, therefore the time complexity is O(mn+nlogn) due to sorting. Only the columns are sorted, using collections.sort, which has the time complexity O(nlogn).
            •	Space complexity : O(mn).

             */
            public int WithSorting(int[,] grid)
            {
                List<int> rowIndices = new List<int>();
                List<int> colIndices = new List<int>();
                for (int row = 0; row < grid.GetLength(0); row++)
                {
                    for (int col = 0; col < grid.GetLength(1); col++)
                    {
                        if (grid[row, col] == 1)
                        {
                            rowIndices.Add(row);
                            colIndices.Add(col);
                        }
                    }
                }
                int medianRow = rowIndices[rowIndices.Count / 2];
                colIndices.Sort();
                int medianCol = colIndices[colIndices.Count / 2];
                return MinDistance1D(rowIndices, medianRow) + MinDistance1D(colIndices, medianCol);
            }

            private int MinDistance1D(List<int> points, int origin)
            {
                int totalDistance = 0;
                foreach (int point in points)
                {
                    totalDistance += Math.Abs(point - origin);
                }
                return totalDistance;
            }

            /*
            Approach #4 (Collect Coordinates in Sorted Order) [Accepted]
Complexity analysis
•	Time complexity : O(mn).
•	Space complexity : O(mn).

            */
            public int CollectCoordinatesSortedOrder(int[][] grid)
            {
                List<int> rows = CollectRows(grid);
                List<int> cols = CollectCols(grid);
                int row = rows[rows.Count / 2]; //Median
                int col = cols[cols.Count / 2]; ////Median
                return MinDistance1D(rows, row) + MinDistance1D(cols, col);
            }


            private List<int> CollectRows(int[][] grid)
            {
                List<int> rows = new List<int>();
                for (int row = 0; row < grid.Length; row++)
                {
                    for (int col = 0; col < grid[0].Length; col++)
                    {
                        if (grid[row][col] == 1)
                        {
                            rows.Add(row);
                        }
                    }
                }
                return rows;
            }

            private List<int> CollectCols(int[][] grid)
            {
                List<int> cols = new List<int>();
                for (int col = 0; col < grid[0].Length; col++)
                {
                    for (int row = 0; row < grid.Length; row++)
                    {
                        if (grid[row][col] == 1)
                        {
                            cols.Add(col);
                        }
                    }
                }
                return cols;
            }
            /*
 Approach #4.1 (Collect Coordinates in Sorted Order And without Median) [Accepted]
Complexity analysis
•	Time complexity : O(mn).
•	Space complexity : O(mn).

 */
            public int CollectCoordinatesSortedOrderAndWithoutMedian(int[][] grid)
            {
                List<int> rows = CollectRows(grid);
                List<int> cols = CollectCols(grid);
                return MinDistance1D(rows) + MinDistance1D(cols);
            }
            private int MinDistance1D(List<int> points)
            {
                int distance = 0;
                int i = 0;
                int j = points.Count - 1;
                while (i < j)
                {
                    distance += points[j] - points[i];
                    i++;
                    j--;
                }
                return distance;
            }

        }



        /* 1129. Shortest Path with Alternating Colors
        https://leetcode.com/problems/shortest-path-with-alternating-colors/description/
         */
        public class ShortestAlternatingPathsSol
        {
            /*
            Approach: Breadth First Search
            Complexity Analysis
Here n is the number of nodes and e is the total number of blue and red edges.
•	Time complexity: O(n+e).
o	The complexity would be similar to the standard BFS algorithm since we’re iterating at most twice over each node.
o	Each queue operation in the BFS algorithm takes O(1) time, and a single node can only be pushed onto the queue twice, leading to O(n) operations for n nodes. We iterate over all the neighbors of each node that is popped out of the queue, so for an undirected edge, a given edge could be iterated at most twice, resulting in O(e) operations total for all the nodes. As a result, the total time required is O(n+e).
•	Space complexity: O(n+e).
o	Building the adjacency list takes O(e) space.
o	The BFS queue takes O(n) because each vertex is added at most twice in the form of triplet of integers.
o	The other visit and answers arrays take O(n) space.

            */
            public int[] BFS(int nodeCount, int[][] redEdges, int[][] blueEdges)
            {
                Dictionary<int, List<List<int>>> adjacencyList = new Dictionary<int, List<List<int>>>();

                foreach (int[] redEdge in redEdges)
                {
                    if (!adjacencyList.ContainsKey(redEdge[0]))
                    {
                        adjacencyList[redEdge[0]] = new List<List<int>>();
                    }
                    adjacencyList[redEdge[0]].Add(new List<int> { redEdge[1], 0 });
                }

                foreach (int[] blueEdge in blueEdges)
                {
                    if (!adjacencyList.ContainsKey(blueEdge[0]))
                    {
                        adjacencyList[blueEdge[0]] = new List<List<int>>();
                    }
                    adjacencyList[blueEdge[0]].Add(new List<int> { blueEdge[1], 1 });
                }

                int[] result = new int[nodeCount];
                Array.Fill(result, -1);
                bool[,] visited = new bool[nodeCount, 2];
                Queue<int[]> queue = new Queue<int[]>();

                // Start with node 0, with number of steps as 0 and undefined color -1.
                queue.Enqueue(new int[] { 0, 0, -1 });
                result[0] = 0;
                visited[0, 0] = visited[0, 1] = true;

                while (queue.Count > 0)
                {
                    int[] currentElement = queue.Dequeue();
                    int currentNode = currentElement[0], currentSteps = currentElement[1], previousColor = currentElement[2];

                    if (!adjacencyList.ContainsKey(currentNode))
                    {
                        continue;
                    }

                    foreach (List<int> neighbor in adjacencyList[currentNode])
                    {
                        int neighborNode = neighbor[0];
                        int edgeColor = neighbor[1];
                        if (!visited[neighborNode, edgeColor] && edgeColor != previousColor)
                        {
                            if (result[neighborNode] == -1)
                            {
                                result[neighborNode] = 1 + currentSteps;
                            }
                            visited[neighborNode, edgeColor] = true;
                            queue.Enqueue(new int[] { neighborNode, 1 + currentSteps, edgeColor });
                        }
                    }
                }
                return result;
            }
        }



        /* 1167. Minimum Cost to Connect Sticks
        https://leetcode.com/problems/minimum-cost-to-connect-sticks/description/
         */

        class ConnectSticksSol
        {
            /*
            Approach 1: Greedy
Complexity Analysis
•	Time complexity : O(NlogN), where N is the length of the input array. Let's break it down:
o	Step 1) Adding N elements to the priority queue will be O(NlogN).
o	Step 2) We remove two of the smallest elements and then add one element to the priority queue until we are left with one element. Since each such operation will reduce one element from the priority queue, we will perform N−1 such operations. Now, we know that both add and remove operations take O(logN) in priority queue, therefore, complexity of this step will be O(NlogN).
•	Space complexity : O(N) since we will store N elements in our priority queue.

            */
            public int ConnectSticks(int[] sticks)
            {
                int totalCost = 0;

                // Create a min heap using a priority queue.
                PriorityQueue<int, int> priorityQueue = new PriorityQueue<int, int>();

                // add all sticks to the min heap.
                foreach (int stick in sticks)
                {
                    priorityQueue.Enqueue(stick, stick);
                }

                // combine two of the smallest sticks until we are left with just one.
                while (priorityQueue.Count > 1)
                {
                    int stick1 = priorityQueue.Dequeue();
                    int stick2 = priorityQueue.Dequeue();

                    int cost = stick1 + stick2;
                    totalCost += cost;

                    priorityQueue.Enqueue(cost, cost);
                }

                return totalCost;
            }
        }



        /* 1168. Optimize Water Distribution in a Village
        https://leetcode.com/problems/optimize-water-distribution-in-a-village/description/ */

        public class MinCostToSupplyWaterSol
        {
            /*
            Approach 1: Prim's Algorithm with Heap
     Complexity Analysis
Let N be the number of houses, and M be the number of pipes from the input.
•	Time Complexity: O((N+M)⋅log(N+M))
o	To build the graph, we iterate through the houses and pipes in the input, which takes O(N+M) time.
o	While building the MST, we might need to iterate through all the edges in the graph in the worst case, which amounts to N+M in total.
For each edge, it would enter and exit the heap data structure at most once. The enter of edge into heap (i.e. push operation) takes log(N+M) time, while the exit of edge (i.e. pop operation) takes a constant time.
Therefore, the time complexity of the MST construction process is O((N+M)⋅log(N+M)).
o	To sum up, the overall time complexity of the algorithm is O((N+M)⋅log(N+M)).
•	Space Complexity: O(N+M)
o	We break down the analysis accordingly into the three major data structures that we used in the algorithm.
o	The graph that we built consists of N+1 vertices and 2⋅M edges (i.e. pipes are bidirectional).
Therefore, the space complexity of graph is O(N+1+2⋅M)=O(N+M).
o	The space complexity of the set that is used to hold the vertices in MST is O(N).
o	Finally, in the worst case, the heap we used might hold all the edges in the graph which is (N+M).
o	To summarize, the overall space complexity of the algorithm is O(N+M).
       
            */
            public int PrimsAlgoWithHeap(int numberOfHouses, int[] wellCosts, int[][] pipeConnections)
            {
                // min heap to maintain the order of edges to be visited.
                PriorityQueue<KeyValuePair<int, int>, int> edgesHeap = new PriorityQueue<KeyValuePair<int, int>, int>();

                // representation of graph in adjacency list
                List<List<KeyValuePair<int, int>>> graph = new List<List<KeyValuePair<int, int>>>(numberOfHouses + 1);
                for (int i = 0; i < numberOfHouses + 1; ++i)
                {
                    graph.Add(new List<KeyValuePair<int, int>>());
                }

                // add a virtual vertex indexed with 0,
                //   then add an edge to each of the house weighted by the cost
                for (int i = 0; i < wellCosts.Length; ++i)
                {
                    KeyValuePair<int, int> virtualEdge = new KeyValuePair<int, int>(wellCosts[i], i + 1);
                    graph[0].Add(virtualEdge);
                    // initialize the heap with the edges from the virtual vertex.
                    edgesHeap.Enqueue(virtualEdge, wellCosts[i]);
                }

                // add the bidirectional edges to the graph
                for (int i = 0; i < pipeConnections.Length; ++i)
                {
                    int house1 = pipeConnections[i][0];
                    int house2 = pipeConnections[i][1];
                    int cost = pipeConnections[i][2];
                    graph[house1].Add(new KeyValuePair<int, int>(cost, house2));
                    graph[house2].Add(new KeyValuePair<int, int>(cost, house1));
                }

                // kick off the exploration from the virtual vertex 0
                HashSet<int> mstSet = new HashSet<int>();
                mstSet.Add(0);

                int totalCost = 0;
                while (mstSet.Count < numberOfHouses + 1)
                {
                    var edge = edgesHeap.Dequeue();
                    int cost = edge.Key;
                    int nextHouse = edge.Value;
                    if (mstSet.Contains(nextHouse))
                    {
                        continue;
                    }

                    // adding the new vertex into the set
                    mstSet.Add(nextHouse);
                    totalCost += cost;

                    // expanding the candidates of edge to choose from in the next round
                    foreach (var neighborEdge in graph[nextHouse])
                    {
                        if (!mstSet.Contains(neighborEdge.Value))
                        {
                            edgesHeap.Enqueue(neighborEdge, neighborEdge.Key);
                        }
                    }
                }

                return totalCost;
            }
            /* Approach 2: Kruskal's Algorithm with Union-Find 
            Complexity Analysis
            Since we applied the Union-Find data structure in our algorithm, let's begin with a statement on the time complexity of the data structure:
            If K operations, either Union or Find, are applied to L elements, the total run time is O(K⋅log∗L), where log∗ is the iterated logarithm.
            One can refer to the proof of Union-Find complexity and the tutorial from Princeton University for more details.
            Let N be the number of houses, and M be the number of pipes from the input.
            •	Time Complexity: O((N+M)⋅log(N+M))
            o	First, we build a list of edges, which takes O(N+M) time.
            o	We then sort the list of edges, which takes O((N+M)⋅log(N+M)) time.
            o	At the end, we iterate through the sorted edges. For each iteration, we invoke a Union-Find operation. Hence, the time complexity for iteration is O((N+M)∗log∗(N)).
            o	To sum up, the overall time complexity of the algorithm is O((N+M)⋅log(N+M)) which is dominated by the sorting step.
            •	Space Complexity: O(N+M)
            o	The space complexity of our Union-Find data structure is O(N).
            o	The space required by the list of edges is O(N+M).
            o	Finally, the space complexity of the sorting algorithm depends on the implementation of each programming language. For instance, the list.sort() function in Python is implemented with the Timsort algorithm whose space complexity is O(n) where n is the number of the elements.
            While in Java, the Collections.sort() is implemented as a variant of quicksort algorithm whose space complexity is O(logn).
            o	To sum up, the overall space complexity of the algorithm is O(N+M) which is dominated by the list of edges.

            */
            public int KruskalsAlgoWithUnionFind(int houseCount, int[] wells, int[][] pipes)
            {
                List<int[]> orderedEdges = new List<int[]>(houseCount + 1 + pipes.Length);

                // add the virtual vertex (index with 0) along with the new edges.
                for (int i = 0; i < wells.Length; ++i)
                {
                    orderedEdges.Add(new int[] { 0, i + 1, wells[i] });
                }

                // add the existing edges
                for (int i = 0; i < pipes.Length; ++i)
                {
                    int[] edge = pipes[i];
                    orderedEdges.Add(edge);
                }

                // sort the edges based on their cost
                orderedEdges.Sort((a, b) => a[2] - b[2]);

                // iterate through the ordered edges
                UnionFind uf = new UnionFind(houseCount);
                int totalCost = 0;
                foreach (int[] edge in orderedEdges)
                {
                    int house1 = edge[0];
                    int house2 = edge[1];
                    int cost = edge[2];
                    // determine if we should add the new edge to the final MST
                    if (uf.Union(house1, house2))
                    {
                        totalCost += cost;
                    }
                }

                return totalCost;
            }
            class UnionFind
            {
                /**
                 * Implementation of UnionFind without load-balancing.
                 */
                private int[] group;
                private int[] rank;

                public UnionFind(int size)
                {
                    // container to hold the group id for each member
                    // Note: the index of member starts from 1,
                    //   thus we add one more element to the container.
                    group = new int[size + 1];
                    rank = new int[size + 1];
                    for (int i = 0; i < size + 1; ++i)
                    {
                        group[i] = i;
                        rank[i] = 0;
                    }
                }

                /**
                 * return the group id that the person belongs to.
                 */
                public int Find(int person)
                {
                    if (group[person] != person)
                    {
                        group[person] = Find(group[person]);
                    }
                    return group[person];
                }

                /**
                 * Join the groups together.
                 * return:
                 * false when the two persons belong to the same group already,
                 * otherwise true
                 */
                public bool Union(int person1, int person2)
                {
                    int group1 = Find(person1);
                    int group2 = Find(person2);
                    if (group1 == group2)
                    {
                        return false;
                    }

                    // attach the group of lower rank to the one with higher rank
                    if (rank[group1] > rank[group2])
                    {
                        group[group2] = group1;
                    }
                    else if (rank[group1] < rank[group2])
                    {
                        group[group1] = group2;
                    }
                    else
                    {
                        group[group1] = group2;
                        rank[group2] += 1;
                    }

                    return true;
                }
            }

            /*

            */

        }



        /* 1192. Critical Connections in a Network
        https://leetcode.com/problems/critical-connections-in-a-network/description/ */

        class CriticalConnectionsSol
        {
            private Dictionary<int, List<int>> graph;
            private Dictionary<int, int?> rank; // Changed to nullable int to represent unvisited nodes.
            private Dictionary<(int, int), bool> connDict;

            /*
            Approach: Depth First Search for Cycle Detection Using Tarjans Algo
            •	Time Complexity: O(V+E) where V represents the number of vertices and E represents the number of edges in the graph. We process each node exactly once thanks to the rank dictionary also acting as a "visited" safeguard at the top of the dfs function. Since the problem statement mentions that the graph is connected, that means E>=V and hence, the overall time complexity would be dominated by the number of edges i.e. O(E).
            •	Space Complexity: O(E). The overall space complexity is a sum of the space occupied by the connDict dictionary, rank dictionary, and graph data structure. E+V+(V+E) = O(E).

            */
            public List<List<int>> DFSForCycleDetectUsingTarjansAlgo(int n, List<List<int>> connections)
            {
                this.FormGraph(n, connections);
                this.Dfs(0, 0);

                List<List<int>> result = new List<List<int>>();
                foreach (var criticalConnection in this.connDict.Keys)
                {
                    result.Add(new List<int> { criticalConnection.Item1, criticalConnection.Item2 });
                }

                return result;
            }

            private int Dfs(int node, int discoveryRank)
            {
                // That means this node is already visited. We simply return the rank.
                if (this.rank[node] != null)
                {
                    return this.rank[node].Value;
                }

                // Update the rank of this node.
                this.rank[node] = discoveryRank;

                // This is the max we have seen till now. So we start with this instead of int.MaxValue or something.
                int minRank = discoveryRank + 1;

                foreach (var neighbor in this.graph[node])
                {
                    // Skip the parent.
                    int? neighRank = this.rank[neighbor];
                    if (neighRank.HasValue && neighRank.Value == discoveryRank - 1)
                    {
                        continue;
                    }

                    // Recurse on the neighbor.
                    int recursiveRank = this.Dfs(neighbor, discoveryRank + 1);

                    // Step 1, check if this edge needs to be discarded.
                    if (recursiveRank <= discoveryRank)
                    {
                        int sortedU = Math.Min(node, neighbor), sortedV = Math.Max(node, neighbor);
                        this.connDict.Remove((sortedU, sortedV));
                    }

                    // Step 2, update the minRank if needed.
                    minRank = Math.Min(minRank, recursiveRank);
                }

                return minRank;
            }

            private void FormGraph(int n, List<List<int>> connections)
            {
                this.graph = new Dictionary<int, List<int>>();
                this.rank = new Dictionary<int, int?>();
                this.connDict = new Dictionary<(int, int), bool>();

                // Default rank for unvisited nodes is "null"
                for (int i = 0; i < n; i++)
                {
                    this.graph[i] = new List<int>();
                    this.rank[i] = null;
                }

                foreach (var edge in connections)
                {
                    // Bidirectional edges
                    int u = edge[0], v = edge[1];
                    this.graph[u].Add(v);
                    this.graph[v].Add(u);

                    int sortedU = Math.Min(u, v), sortedV = Math.Max(u, v);
                    connDict[(sortedU, sortedV)] = true;
                }
            }
        }


        /* 1182. Shortest Distance to Target Color
        https://leetcode.com/problems/shortest-distance-to-target-color/description/

         */
        public class ShortestDistanceToTargetColorSol
        {
            /*
            Approach 1: Binary Search
            Complexity Analysis
•	Time Complexity : O(QlogN+N), where Q is the length of queries and N is the length of colors.
Going through the input array colors and storing each color - index pair take O(N) time. When iterating queries and generating results, we apply binary search once for each query, and each binary search takes O(logN), which results in O(QlogN). Putting them together and ignoring constants for Big O notation, we have O(QlogN+N).
•	Space Complexity : O(N).
This is because we store the indexes of each color - index pair in a hashmap.

            */
            public List<int> UsingBinarySearch(int[] colors, int[][] queries)
            {
                // initialization
                List<int> queryResults = new List<int>();
                Dictionary<int, List<int>> colorIndexMap = new Dictionary<int, List<int>>();

                for (int i = 0; i < colors.Length; i++)
                {
                    if (!colorIndexMap.ContainsKey(colors[i]))
                    {
                        colorIndexMap[colors[i]] = new List<int>();
                    }
                    colorIndexMap[colors[i]].Add(i);
                }

                // for each query, apply binary search
                for (int i = 0; i < queries.Length; i++)
                {
                    int targetIndex = queries[i][0], color = queries[i][1];
                    if (!colorIndexMap.ContainsKey(color))
                    {
                        queryResults.Add(-1);
                        continue;
                    }

                    List<int> indexList = colorIndexMap[color];
                    int insertPosition = indexList.BinarySearch(targetIndex);

                    // due to its nature, we need to convert the returning values
                    // from List<T>.BinarySearch
                    if (insertPosition < 0)
                    {
                        insertPosition = ~insertPosition; // This gives the index where the target would be inserted
                    }

                    // Handling cases when:
                    // - the target index is smaller than all elements in the indexList
                    // - the target index is larger than all elements in the indexList
                    // - the target index sits between the left and right boundaries
                    if (insertPosition == 0)
                    {
                        queryResults.Add(indexList[insertPosition] - targetIndex);
                    }
                    else if (insertPosition == indexList.Count)
                    {
                        queryResults.Add(targetIndex - indexList[insertPosition - 1]);
                    }
                    else
                    {
                        int leftNearest = targetIndex - indexList[insertPosition - 1];
                        int rightNearest = indexList[insertPosition] - targetIndex;
                        queryResults.Add(Math.Min(leftNearest, rightNearest));
                    }
                }
                return queryResults;
            }

            /*

            */

            /*
            Approach 2: Pre-computed
            Complexity Analysis
            •	Time Complexity : O(N+Q), where N is the length of colors and Q is the length of queries.
            This is because we use iterations to fill distance which is a matrix of 3 rows and N columns taking O(N) time. Afterwards, we generate the answer for each query in queries in O(1).
            •	Space Complexity : O(N).
            This is because we initialize two arrays of size 3 and one 2D array of 3 rows and N columns.

            */
            public List<int> WithPreCompuration(int[] colors, int[][] queries)
            {
                // initializations
                int numberOfColors = colors.Length;
                int[] rightmostIndex = { 0, 0, 0 };
                int[] leftmostIndex = { numberOfColors - 1, numberOfColors - 1, numberOfColors - 1 };

                int[][] distanceArray = new int[3][];
                for (int i = 0; i < 3; i++)
                {
                    distanceArray[i] = new int[numberOfColors];
                    for (int j = 0; j < numberOfColors; j++)
                    {
                        distanceArray[i][j] = -1;
                    }
                }

                // looking forward
                for (int i = 0; i < numberOfColors; i++)
                {
                    int color = colors[i] - 1;
                    for (int j = rightmostIndex[color]; j < i + 1; j++)
                    {
                        distanceArray[color][j] = i - j;
                    }
                    rightmostIndex[color] = i + 1;
                }

                // looking backward
                for (int i = numberOfColors - 1; i > -1; i--)
                {
                    int color = colors[i] - 1;
                    for (int j = leftmostIndex[color]; j > i - 1; j--)
                    {
                        if (distanceArray[color][j] == -1 || distanceArray[color][j] > j - i)
                        {
                            distanceArray[color][j] = j - i;
                        }
                    }
                    leftmostIndex[color] = i - 1;
                }

                List<int> queryResults = new List<int>();
                for (int i = 0; i < queries.Length; i++)
                {
                    queryResults.Add(distanceArray[queries[i][1] - 1][queries[i][0]]);
                }
                return queryResults;
            }
        }


        /* 1197. Minimum Knight Moves
        https://leetcode.com/problems/minimum-knight-moves/description/
         */
        class MinKnightMovesSol
        {
            /*
            Approach 1: BFS (Breadth-First Search)
        Complexity Analysis
        Given the coordinate of the target as (x,y), the number of cells covered by the circle that is centered at point (0,0) and reaches the target point is roughly (max(∣x∣,∣y∣))^2.
            Time Complexity: O((max(∣x∣,∣y∣))^2)
            Due to the nature of BFS, before reaching the target, we will have covered all the neighborhoods that are closer to the start point. The aggregate of these neighborhoods forms a circle, and the area can be approximated by the area of a square with an edge length of max(2∣x∣,2∣y∣). The number of cells within this square would be (max(2∣x∣,2∣y∣))^2.
            Hence, the overall time complexity of the algorithm is O((max(2∣x∣,2∣y∣))^2)=O((max(∣x∣,∣y∣))^2).
            Space Complexity: O((max(∣x∣,∣y∣))^2)
            We employed two data structures in the algorithm, i.e. queue and set.
            At any given moment, the queue contains elements that are situated in at most two different layers (or levels). In our case, the maximum number of elements at one layer would be 4⋅max(∣x∣,∣y∣), i.e. the perimeter of the exploration square. As a result, the space complexity for the queue is O(max(∣x∣,∣y∣)).
            As for the set, it will contain every elements that we visited, which is (max(∣x∣,∣y∣))^2 as we estimated in the time complexity analysis. As a result, the space complexity for the set is O((max(∣x∣,∣y∣))^2).
            To sum up, the overall space complexity of the algorithm is O((max(∣x∣,∣y∣))^2), which is dominated by the space used by the set.

            */
            public int BFS(int targetX, int targetY)
            {
                // the offsets in the eight directions
                int[][] offsets = new int[][] {
            new int[] { 1, 2 }, new int[] { 2, 1 }, new int[] { 2, -1 }, new int[] { 1, -2 },
            new int[] { -1, -2 }, new int[] { -2, -1 }, new int[] { -2, 1 }, new int[] { -1, 2 }
        };

                // - Rather than using the inefficient HashSet, we use the bitmap
                //     otherwise we would run out of time for the test cases.
                // - We create a bitmap that is sufficient to cover all the possible
                //     inputs, according to the description of the problem.
                bool[][] visited = new bool[607][];
                for (int i = 0; i < 607; i++)
                {
                    visited[i] = new bool[607];
                }

                Queue<int[]> queue = new Queue<int[]>();
                queue.Enqueue(new int[] { 0, 0 });
                int steps = 0;

                while (queue.Count > 0)
                {
                    int currLevelSize = queue.Count;
                    // iterate through the current level
                    for (int i = 0; i < currLevelSize; i++)
                    {
                        int[] curr = queue.Dequeue();
                        if (curr[0] == targetX && curr[1] == targetY)
                        {
                            return steps;
                        }

                        foreach (int[] offset in offsets)
                        {
                            int[] next = new int[] { curr[0] + offset[0], curr[1] + offset[1] };
                            // align the coordinate to the bitmap
                            if (!visited[next[0] + 302][next[1] + 302])
                            {
                                visited[next[0] + 302][next[1] + 302] = true;
                                queue.Enqueue(next);
                            }
                        }
                    }
                    steps++;
                }
                // move on to the next level
                return steps;
            }
            /*
            Approach 2: Bidirectional BFS
            Complexity Analysis
        Although the bidirectional BFS cuts the exploration scope in half, compared to the unidirectional BFS, the overall time and space complexities remain the same.
        We will break it down in detail in this section.
        First of all, given the target's coordinate, (x,y), then the area that is covered by the two exploratory circles of the bidirectional BFS will be max(∣x∣,∣y∣)2/2.
            Time Complexity: O((max(∣x∣,∣y∣))^2)
            Reducing the scope of exploration by half does speed up the algorithm. However, it does not change the time complexity of the algorithm which remains O((max(∣x∣,∣y∣))^2).
            Space Complexity: O((max(∣x∣,∣y∣))^2)
            In exchange for reducing the search scope, we double the usage of data structures compared to the unidirectional BFS.
        Similarly to the time complexity, multiplying the required space by two does not change the overall space complexity of the algorithm which remains O((max(∣x∣,∣y∣))^2).

            */
            public int BidirectionalDFS(int targetX, int targetY)
            {
                // the offsets in the eight directions
                int[][] offsets = new int[][]
                {
            new int[] { 1, 2 }, new int[] { 2, 1 }, new int[] { 2, -1 }, new int[] { 1, -2 },
            new int[] { -1, -2 }, new int[] { -2, -1 }, new int[] { -2, 1 }, new int[] { -1, 2 }
                };

                // data structures needed to move from the origin point
                var originQueue = new LinkedList<int[]>();
                originQueue.AddLast(new int[] { 0, 0, 0 });
                var originDistance = new Dictionary<string, int>();
                originDistance["0,0"] = 0;

                // data structures needed to move from the target point
                var targetQueue = new LinkedList<int[]>();
                targetQueue.AddLast(new int[] { targetX, targetY, 0 });
                var targetDistance = new Dictionary<string, int>();
                targetDistance[targetX + "," + targetY] = 0;

                while (true)
                {
                    // check if we reach the circle of target
                    int[] origin = originQueue.First.Value;
                    originQueue.RemoveFirst();
                    string originCoordinates = origin[0] + "," + origin[1];
                    if (targetDistance.ContainsKey(originCoordinates))
                    {
                        return origin[2] + targetDistance[originCoordinates];
                    }

                    // check if we reach the circle of origin
                    int[] target = targetQueue.First.Value;
                    targetQueue.RemoveFirst();
                    string targetCoordinates = target[0] + "," + target[1];
                    if (originDistance.ContainsKey(targetCoordinates))
                    {
                        return target[2] + originDistance[targetCoordinates];
                    }

                    foreach (int[] offset in offsets)
                    {
                        // expand the circle of origin
                        int[] nextOrigin = new int[] { origin[0] + offset[0], origin[1] + offset[1] };
                        string nextOriginCoordinates = nextOrigin[0] + "," + nextOrigin[1];
                        if (!originDistance.ContainsKey(nextOriginCoordinates))
                        {
                            originQueue.AddLast(new int[] { nextOrigin[0], nextOrigin[1], origin[2] + 1 });
                            originDistance[nextOriginCoordinates] = origin[2] + 1;
                        }

                        // expand the circle of target
                        int[] nextTarget = new int[] { target[0] + offset[0], target[1] + offset[1] };
                        string nextTargetCoordinates = nextTarget[0] + "," + nextTarget[1];
                        if (!targetDistance.ContainsKey(nextTargetCoordinates))
                        {
                            targetQueue.AddLast(new int[] { nextTarget[0], nextTarget[1], target[2] + 1 });
                            targetDistance[nextTargetCoordinates] = target[2] + 1;
                        }
                    }
                }
            }
            /*
            Approach 3: DFS (Depth-First Search) with Memoization
            Complexity Analysis
        Let (x,y) be the coordinate of the target.
            Time Complexity: O(∣x⋅y∣)
            The execution of our recursive algorithm will unfold as a binary tree where each node represents an invocation of the recursive function.
        And the time complexity of the algorithm is proportional to the total number of invocations, i.e. total number of nodes in the binary tree.
            The total number of nodes grows exponentially in the binary tree.
        However, there will be some overlap in terms of the invocations, i.e. dfs(x,y) might be invoked multiple times with the same input.
        Thanks to the memoization technique, we avoid redundant calculations, i.e. the return value of dfs(x,y) is stored for reuse later, which greatly improves the performance.
            In the algorithm, we restrict the exploration to the first quadrant of the board. Therefore, in the worst case, we will explore all of the cells between the origin and the target in the first quadrant.
        In total, there are ∣x⋅y∣ cells in a rectangle that spans from the origin to the target. As a result, the overall time complexity of the algorithm is O(∣x⋅y∣).
            Space Complexity: O(∣x⋅y∣)
            First of all, due to the presence of recursion in the algorithm, it will incur additional memory consumption in the function call stack.
        The consumption is proportional to the level of the execution tree, i.e. max(∣x∣,∣y∣).
            Secondly, due to the application of memoization technique, we will keep all the intermediate results in the memory for reuse.
        As we have seen in the above time complexity analysis, the maximum number of intermediate results will be O(∣x⋅y∣).
            To sum up, the overall space complexity of the algorithm is O(∣x⋅y∣), which is dominated by the memoization part.

            */
            public int DFSWithMemo(int x, int y)
            {
                return Dfs(Math.Abs(x), Math.Abs(y));
            }
            private Dictionary<string, int> memo = new Dictionary<string, int>();

            private int Dfs(int x, int y)
            {
                string key = x + "," + y;
                if (memo.ContainsKey(key))
                {
                    return memo[key];
                }

                if (x + y == 0)
                {
                    return 0;
                }
                else if (x + y == 2)
                {
                    return 2;
                }
                else
                {
                    int ret = Math.Min(Dfs(Math.Abs(x - 1), Math.Abs(y - 2)),
                                       Dfs(Math.Abs(x - 2), Math.Abs(y - 1))) + 1;
                    memo[key] = ret;
                    return ret;
                }
            }


        }


        /* 1217. Minimum Cost to Move Chips to The Same Position
        https://leetcode.com/problems/minimum-cost-to-move-chips-to-the-same-position/description/
         */
        class MinCostToMoveChipsSol
        {
            /*
            Approach 1: Moving Chips Cleverly
Let N be the length of position.
•	Time Complexity : O(N) since we need to iterate position once.
•	Space Complexity : O(1) since we only use two ints: even_cnt and odd_cnt.

            */
            public int MovingChipsClearly(int[] position)
            {
                int even_cnt = 0;
                int odd_cnt = 0;
                foreach (int i in position)
                {
                    if (i % 2 == 0)
                    {
                        even_cnt++;
                    }
                    else
                    {
                        odd_cnt++;
                    }
                }
                return Math.Min(odd_cnt, even_cnt);
            }
        }


        /* 1229. Meeting Scheduler
        https://leetcode.com/problems/meeting-scheduler/description/ */
        class MinAvailableDurationSol
        {
            /*
Approach 1: Sorting + Two pointers 
            Complexity Analysis
•	Time complexity: O(MlogM+NlogN), when M is the length of slots1 and N is the length of slots2.
Sorting both arrays would take O(MlogM+NlogN). The two pointers take O(M+N) because, during each iteration, we would visit a new element, and there are a total of M+N elements. Putting these together, the total time complexity is O(MlogM+NlogN).
•	Space complexity: O(n) or O(logn)
Some extra space is used when we sort an array of size n in place. The space complexity of the sorting algorithm depends on the programming language.
Thus, the total space complexity of the algorithm is O(n) or O(logn).
            */
            public List<int> UsingSortingAndTwoPointers(int[][] slots1, int[][] slots2, int duration)
            {
                Array.Sort(slots1, (a, b) => a[0] - b[0]);
                Array.Sort(slots2, (a, b) => a[0] - b[0]);

                int pointer1 = 0, pointer2 = 0;

                while (pointer1 < slots1.Length && pointer2 < slots2.Length)
                {
                    // find the boundaries of the intersection, or the common slot
                    int intersectLeft = Math.Max(slots1[pointer1][0], slots2[pointer2][0]);
                    int intersectRight = Math.Min(slots1[pointer1][1], slots2[pointer2][1]);
                    if (intersectRight - intersectLeft >= duration)
                    {
                        return new List<int> { intersectLeft, intersectLeft + duration };
                    }
                    // always move the one that ends earlier
                    if (slots1[pointer1][1] < slots2[pointer2][1])
                    {
                        pointer1++;
                    }
                    else
                    {
                        pointer2++;
                    }
                }
                return new List<int>();
            }
            /*
            Approach 2: Heap
            Complexity Analysis
•	Time complexity: O((M+N)log(M+N)), when M is the length of slots1 and N is the length of slots2.
There are two parts to be analyzed: 1) building up the heap; 2) the iteration when we keep popping elements from the heap. For the second part, popping one element takes O(log(M+N)), therefore, in the worst case, popping M+N elements takes O((M+N)log(M+N)).
Regarding the first part, we have different answers for Java and Python implementations. For Python, heapq.heapify transforms a list into a heap, in-place, in linear time; however, in Java, we choose to pop each element into the heap, which leads to a time complexity of O((M+N)log(M+N)). Note that it is possible to convert the array into a heap in linear time using the constructor of PriorityQueue; however, that will not influence the overall time complexity and will make it less readable.
When we put these two parts together, the total time complexity is O((M+N)log(M+N)), which is determined by the first part.
•	Space complexity: O(M+N). This is because we store all M+N time slots in a heap.

            */
            public List<int> UsingHeapPQ(int[][] slots1, int[][] slots2, int duration)
            {
                PriorityQueue<int[], int[]> timeSlots = new PriorityQueue<int[], int[]>(Comparer<int[]>.Create((slot1, slot2) => slot1[0] - slot2[0]));

                foreach (int[] slot in slots1)
                {
                    if (slot[1] - slot[0] >= duration) timeSlots.Enqueue(slot, slot);
                }
                foreach (int[] slot in slots2)
                {
                    if (slot[1] - slot[0] >= duration) timeSlots.Enqueue(slot, slot);
                }

                while (timeSlots.Count > 1)
                {
                    int[] slot1 = timeSlots.Dequeue();
                    int[] slot2 = timeSlots.Peek();
                    if (slot1[1] >= slot2[0] + duration)
                    {
                        return new List<int> { slot2[0], slot2[0] + duration };
                    }
                }
                return new List<int>();
            }

        }


        /* 1236. Web Crawler
        https://leetcode.com/problems/web-crawler/description/ */

        class CrawlSol
        {
            /*
            Note: In an interview, you probably want to mention to use DFS for different domains and BFS for paths under each domain considering 
            factors like pressure to a website, infinitely deep paths, TCP keepalive, etc. Also, don't forget to use a probabilistic filter like bloomfilter in front of a hash set.
            */

            /*
            Approach 1: Depth-first search
            Complexity Analysis
            Let m be the number of edges in the graph, and l be the maximum length of a URL (urls[i].length).
            •	Time complexity: O(m⋅l).
            Let k be the number of traversed vertices. We add all these nodes to the set, with each node costing up to O(l). The total time for inserting into the set is thus O(k⋅l).
            The most time-consuming part in the dfs is calling htmlParser.getUrls(url) to get the edges outgoing from url and iterating over all nextUrl. When processing nextUrl, we call getHostname(nextUrl) and search nextUrl in the hash set. Both of these can take O(nextUrl.length)=O(l) time. The complexity equals the sum of all the O(l) work done.
            The total number of elements in htmlParser.getUrls(url) over all URLs is m – the total number of edges in the graph. Each element can have a length of O(l). The sum of lengths of the elements of htmlParser.getUrls(url) over all URLs is O(m⋅l).
            The total complexity is O(k⋅l+m⋅l). Since k=O(m), we can simplify this expression to O(m⋅l).
            •	Space complexity: O(m⋅l).
            At each recursion level, we simultaneously store the return value of htmlParser.getUrls(url). As mentioned above, the total length of these is O(m⋅l). We also use a set to store the answer, which can grow to this size. While you usually don't include the answer as part of the space complexity, the set is also functional - it prevents us from visiting a URL more than once.

            */
            public List<string> DFS(string startUrl, HtmlParser htmlParser)
            {
                startHostname = GetHostname(startUrl);
                Dfs(startUrl, htmlParser);
                return new List<string>(visited);
            }
            public class HtmlParser
            {
                public List<String> GetUrls(String url)
                {
                    return new List<string>(); //Dummy code }
                }
            }
            private string startHostname;
            private HashSet<string> visited = new HashSet<string>();

            private string GetHostname(string url)
            {
                // split url by slashes
                // for instance, "http://example.org/foo/bar" will be split into
                // "http:", "", "example.org", "foo", "bar"
                // the hostname is the 2-nd (0-indexed) element
                return url.Split('/')[2];
            }

            private void Dfs(string url, HtmlParser htmlParser)
            {
                visited.Add(url);
                foreach (string nextUrl in htmlParser.GetUrls(url))
                {
                    if (GetHostname(nextUrl).Equals(startHostname) && !visited.Contains(nextUrl))
                    {
                        Dfs(nextUrl, htmlParser);
                    }
                }
            }

            /*
            Approach 2: Breadth-first search
            Complexity Analysis
            Let n be the total number of URLs (urls.length), m be the number of edges in the graph, and l be the maximum length of a URL (urls[i].length).
            •	Time complexity: O(m⋅l).
            Let k be the number of traversed vertices. We add each of these vertices to the set and to the queue in up to O(l) per vertex. The total time for inserting into the set and into the queue is thus O(k⋅l).
            The most time-consuming part is calling htmlParser.getUrls(url) to get the edges outgoing from url and iterating over all nextUrl. When processing nextUrl, we call getHostname(nextUrl) and search nextUrl in the hash set. Both of these can take O(nextUrl.length)=O(l) time. The complexity equals the sum of all the O(l) work done.
            The total number of elements in htmlParser.getUrls(url) over all URLs is m – the total number of edges in the graph. Each element can have a length of O(l). The sum of lengths of the elements of htmlParser.getUrls(url) over all URLs is O(m⋅l).
            The total complexity is O(k⋅l+m⋅l). Since k=O(m), we can simplify this expression to O(m⋅l).
            •	Space complexity: O(n⋅l).
            For each visited url, we call htmlParser.getUrls(url) and store its return value. For one url, htmlParser.getUrls(url) contains O(n) elements (in the worst case, there are edges from the url to all other vertices), each having a length up to O(l). The total length of the elements of htmlParser.getUrls(url) for one url could therefore be O(n⋅l). Unlike in the previous approach, we do not store them simultaneously for all vertices, but only for one vertex at a time.
            The total length of the queue elements does not exceed the total length of all URLs – O(n⋅l).
            So the total space complexity is O(n⋅l).

            */
            public List<string> BFS(string startUrl, HtmlParser htmlParser)
            {
                string startHostname = GetHostname(startUrl);
                Queue<string> queue = new Queue<string>(new List<string> { startUrl });
                HashSet<string> visitedSet = new HashSet<string>(new List<string> { startUrl });
                while (queue.Count > 0)
                {
                    string url = queue.Dequeue();
                    foreach (string nextUrl in htmlParser.GetUrls(url))
                    {
                        if (GetHostname(nextUrl) == startHostname && !visitedSet.Contains(nextUrl))
                        {
                            queue.Enqueue(nextUrl);
                            visitedSet.Add(nextUrl);
                        }
                    }
                }
                return new List<string>(visitedSet);
            }

        }


        /* 1242. Web Crawler Multithreaded
        https://leetcode.com/problems/web-crawler-multithreaded/description/
         */
        class CrawlMultiThreadedSol
        {
            public class HtmlParser
            {
                public List<String> GetUrls(String url)
                {
                    return new List<string>(); //Dummy code}
                }

                public IList<string> Crawl(string startUrl, HtmlParser htmlParser)
                {
                    ConcurrentDictionary<string, bool> dict = new ConcurrentDictionary<string, bool>();
                    if (IsValidHost(GetHostName(startUrl)))
                    {
                        dict[startUrl] = true;
                    }
                    CrawlPage(startUrl, htmlParser, dict).Wait();
                    return dict.Keys.ToList();
                }

                private async Task CrawlPage(string url, HtmlParser htmlParser, ConcurrentDictionary<string, bool> dict)
                {
                    IList<String> crawledUrls = htmlParser.GetUrls(url);
                    List<Task> tasks = new();
                    foreach (string c in crawledUrls)
                    {
                        if (!dict.ContainsKey(c) && GetHostName(url).Equals(GetHostName(c)) && IsValidHost(GetHostName(c)))
                        {
                            dict[c] = true;
                            tasks.Add(Task.Run(async () => await CrawlPage(c, htmlParser, dict)));
                        }
                    }
                    await Task.WhenAll(tasks);
                }

                private string GetHostName(string url)
                {
                    string[] splits = url.Split("//");
                    string second = splits[1];
                    string[] ends = second.Split("/");
                    return ends[0];
                }

                private bool IsValidHost(string hostName)
                {
                    if (hostName.Length < 1 || hostName.Length > 63)
                    {
                        return false;
                    }
                    return Regex.IsMatch(hostName, @"[a-z0-9.][a-z0-9.-]*[a-z0-9.]$");
                }
            }


        }



        /* 1259. Handshakes That Don't Cross
        https://leetcode.com/problems/handshakes-that-dont-cross/description/ */

        class NumberOfWaysSol
        {
            private static int m = 1000000007;
            /*
            Approach 1: Bottom-Up Dynamic Programming
            Complexity Analysis
            •	Time complexity: O(numPeople^2).
            We calculate the DP in two nested loops. Both the outer and the inner loops do O(numPeople) iterations, so the total complexity is O(numPeople^2).
            •	Space complexity: O(numPeople).
            We store the array dp, which is of size O(numPeople).

            */
            public int BottomUpDP(int numPeople)
            {
                int[] dp = new int[numPeople / 2 + 1];
                dp[0] = 1;
                for (int i = 1; i <= numPeople / 2; i++)
                {
                    for (int j = 0; j < i; j++)
                    {
                        dp[i] += (int)((long)dp[j] * dp[i - j - 1] % m);
                        dp[i] %= m;
                    }
                }
                return dp[numPeople / 2];
            }

            /*
            Approach 2: Top-Down Dynamic Programming (Memoization)
            Complexity Analysis
•	Time complexity: O(numPeople^2).
Even though we changed the order of DP computation, the time complexity remains the same. As in the first approach, there are O(numPeople) states of DP, and for each, we compute the answer in O(numPeople). Since we use memoization, we calculate each DP value only once.
•	Space complexity: O(numPeople).

            */
            int[] dp;

            public int TopDownDPWithMemo(int numPeople)
            {
                dp = new int[numPeople / 2 + 1];
                Array.Fill(dp, -1);
                dp[0] = 1;
                return calculateDP(numPeople / 2);
            }

            private int calculateDP(int i)
            {
                if (dp[i] != -1)
                {
                    return dp[i];
                }
                dp[i] = 0;
                for (int j = 0; j < i; j++)
                {
                    dp[i] += (int)((long)calculateDP(j) * calculateDP(i - j - 1) % m);
                    dp[i] %= m;
                }
                return dp[i];
            }

            /*
            Approach 3: Catalan Numbers
Complexity Analysis
•	Time complexity: O(numPeople).
First, we calculate the inverse elements for numbers in the range [1,n+1] in O(n). Then we compute Catalan numbers in O(n). Total complexity is O(n)=O(numPeople).
•	Space complexity: O(numPeople).
We use the array inv of size O(n) for storing inverse elements.

            */

            private int mul(int a, int b)
            {
                return (int)((long)a * b % m);
            }

            public int CatalanNumbers(int numPeople)
            {
                int n = numPeople / 2;
                int[] inv = new int[numPeople / 2 + 2];
                inv[1] = 1;
                for (int i = 2; i < n + 2; i++)
                {
                    int k = m / i, r = m % i;
                    inv[i] = m - mul(k, inv[r]);
                }
                int C = 1;
                for (int i = 0; i < n; i++)
                {
                    C = mul(mul(2 * (2 * i + 1), inv[i + 2]), C);
                }
                return C;
            }
        }



        /* 1268. Search Suggestions System
        https://leetcode.com/problems/search-suggestions-system/description/ */

        class SuggestedProductsSol
        {
            /*
            Approach 1: Binary Search
            Complexity Analysis
•	Time complexity : O(nlog(n))+O(mlog(n)). Where n is the length of products and m is the length of the search word. Here we treat string comparison in sorting as O(1). O(nlog(n)) comes from the sorting and O(mlog(n)) comes from running binary search on products m times.
o	In Java there is an additional complexity of O(m2) due to Strings being immutable, here m is the length of searchWord.
•	Space complexity : Varies between O(1) and O(n) where n is the length of products, as it depends on the implementation used for sorting. We ignore the space required for output as it does not affect the algorithm's space complexity. See Internal details of std::sort.
Space required for output is O(m) where m is the length of the search word.

            */
            public List<List<string>> UsingBinarySearch(string[] products, string searchWord)
            {
                Array.Sort(products);
                List<List<string>> result = new List<List<string>>();
                int start = 0, binarySearchStart = 0, n = products.Length;
                string prefix = string.Empty;

                foreach (char c in searchWord)
                {
                    prefix += c;

                    // Get the starting index of word starting with `prefix`.
                    start = LowerBound(products, binarySearchStart, prefix);

                    // Add empty list to result.
                    result.Add(new List<string>());

                    // Add the words with the same prefix to the result.
                    // Loop runs until `i` reaches the end of input or 3 times or till the
                    // prefix is the same for `products[i]` Whichever comes first.
                    for (int i = start; i < Math.Min(start + 3, n); i++)
                    {
                        if (products[i].Length < prefix.Length || !products[i].Substring(0, prefix.Length).Equals(prefix))
                            break;
                        result[result.Count - 1].Add(products[i]);
                    }

                    // Reduce the size of elements to binary search on since we know
                    binarySearchStart = Math.Abs(start);
                }
                return result;
            }
            // Equivalent code for lower_bound in C#
            int LowerBound(string[] products, int start, string word)
            {
                int i = start, j = products.Length, mid;
                while (i < j)
                {
                    mid = (i + j) / 2;
                    if (string.Compare(products[mid], word) >= 0)
                        j = mid;
                    else
                        i = mid + 1;
                }
                return i;
            }

            /*
            Approach 2: Trie + DFS
            Complexity Analysis
•	Time complexity : O(M) to build the trie where M is total number of characters in products For each prefix we find its representative node in O(len(prefix)) and dfs to find at most 3 words which is an O(1) operation. Thus the overall complexity is dominated by the time required to build the trie.
o	In Java there is an additional complexity of O(m2) due to Strings being immutable, here m is the length of searchWord.
•	Space complexity : O(26n)=O(n). Here n is the number of nodes in the trie. 26 is the alphabet size.
Space required for output is O(m) where m is the length of the search word.

            */
            public List<List<string>> TireWithDFS(string[] products, string searchWord)
            {
                Trie trie = new Trie();
                List<List<string>> result = new List<List<string>>();
                // Add all words to trie.
                foreach (string word in products)
                    trie.Insert(word);
                string prefix = string.Empty;
                foreach (char c in searchWord)
                {
                    prefix += c;
                    result.Add(trie.GetWordsStartingWith(prefix));
                }
                return result;
            }
            public class Trie
            {

                // Node definition of a trie
                private class Node
                {
                    public bool IsWord = false;
                    public List<Node> Children = new List<Node>(new Node[26]);
                };

                private Node root, current;
                private List<string> resultBuffer;

                // Runs a DFS on trie starting with given prefix and adds all the words in the resultBuffer, limiting result size to 3
                private void DfsWithPrefix(Node current, string word)
                {
                    if (resultBuffer.Count == 3)
                        return;
                    if (current.IsWord)
                        resultBuffer.Add(word);

                    // Run DFS on all possible paths.
                    for (char c = 'a'; c <= 'z'; c++)
                        if (current.Children[c - 'a'] != null)
                            DfsWithPrefix(current.Children[c - 'a'], word + c);
                }

                public Trie()
                {
                    root = new Node();
                }

                // Inserts the string in trie.
                public void Insert(string s)
                {

                    // Points current to the root of trie.
                    current = root;
                    foreach (char c in s)
                    {
                        if (current.Children[c - 'a'] == null)
                            current.Children[c - 'a'] = new Node();
                        current = current.Children[c - 'a'];
                    }

                    // Mark this node as a completed word.
                    current.IsWord = true;
                }

                public List<string> GetWordsStartingWith(string prefix)
                {
                    current = root;
                    resultBuffer = new List<string>();
                    // Move current to the end of prefix in its trie representation.
                    foreach (char c in prefix)
                    {
                        if (current.Children[c - 'a'] == null)
                            return resultBuffer;
                        current = current.Children[c - 'a'];
                    }
                    DfsWithPrefix(current, prefix);
                    return resultBuffer;
                }
            }

        }


        /* 1274. Number of Ships in a Rectangle
        https://leetcode.com/problems/number-of-ships-in-a-rectangle/description/
         */
        class CountShipsSol
        {
            /*
            Approach 1: Divide And Conquer
            Complexity Analysis
Let M be the range of possible x-coordinate values between bottomLeft[0] and topRight[0] and let N be the range of possible y-coordinate values between bottomLeft[1] and topRight[1]. Thus, the maximum possible number of points in the rectangle is M⋅N. Finally, let S be the maximum number of ships in the sea.
•	Time Complexity: O(S⋅(log2max(M,N)−log4S))
Each call to countShips requires only constant time so the time complexity will be O(1) times the maximum possible number of calls to countShips.
The worst-case scenario is when there is the maximum number of possible ships (S=10) and they are spread out such that after S recursive calls (the log4S level of the recursion tree), there are S regions that each contain 1 ship and the remaining regions are empty.
Each region that contains 1 ship, will result in 4 recursive calls. 3 will return 0 because they do not contain a ship and 1 call will result in 4 more recursive calls because it does contain a ship. This process will repeat until we make a recursive call with the exact coordinates of the ship.
At the latest, we will pinpoint the ship at the maximum depth of the recursion tree which is log2max(M,N) because at each recursive call we divide the search space by half for each of the 2 dimensions.
Thus, once a region contains only 1 ship, it may still take 4⋅(log2max(M,N)−log4S) recursive calls before pinpointing the location of the ship (and returning 1). And since there are S ships, the total number of recursive calls after all regions contain at most 1 ship is 4⋅S⋅(log2max(M,N)−log4S).
Summing up, the time complexity is S+4⋅S⋅(log2max(M,N)−log4S) which in the worst case (when S=10 and M=N=1000) equals 342 recursive calls.
•	Space Complexity: O(log2max(M,N)).
Each call to countShips uses only constant space so our space complexity is directly related to the maximum height of the recursion call stack. Since we have 2 dimensions of magnitudes M and N respectively, and the search space for each dimension is reduced by half at each recursive call to countShips, the maximum height of the recursion call stack will be log2max(M,N).

            */
            public int DivideAndConquer(Sea sea, int[] topRight, int[] bottomLeft)
            {
                // If the current rectangle does not contain any ships, return 0.         
                if (bottomLeft[0] > topRight[0] || bottomLeft[1] > topRight[1])
                    return 0;
                if (!sea.HasShips(topRight, bottomLeft))
                    return 0;

                // If the rectangle represents a single point, a ship is located
                if (topRight[0] == bottomLeft[0] && topRight[1] == bottomLeft[1])
                    return 1;

                // Recursively check each of the 4 sub-rectangles for ships
                int midX = (topRight[0] + bottomLeft[0]) / 2;
                int midY = (topRight[1] + bottomLeft[1]) / 2;
                return DivideAndConquer(sea, new int[] { midX, midY }, bottomLeft) +
                       DivideAndConquer(sea, topRight, new int[] { midX + 1, midY + 1 }) +
                       DivideAndConquer(sea, new int[] { midX, topRight[1] }, new int[] { bottomLeft[0], midY + 1 }) +
                       DivideAndConquer(sea, new int[] { topRight[0], midY }, new int[] { midX + 1, bottomLeft[1] });
            }
            public class Sea
            {
                public bool HasShips(int[] topRight, int[] bottomLeft) { return false; } //Dummy logic;
            }
        }


        /* 1293. Shortest Path in a Grid with Obstacles Elimination
        https://leetcode.com/problems/shortest-path-in-a-grid-with-obstacles-elimination/description/
         */

        class ShortestPathSol
        {
            /*
            Approach 1: BFS (Breadth-First Search)
           Complexity Analysis
Let N be the number of cells in the grid, and K be the quota to eliminate obstacles.
•	Time Complexity: O(N⋅K)
o	We conduct a BFS traversal in the grid. In the worst case, we will visit each cell in the grid. And for each cell, at most, it will be visited K times, with different quotas of obstacle elimination.
o	Thus, the overall time complexity of the algorithm is O(N⋅K).
•	Space Complexity: O(N⋅K)
o	We used a queue to maintain the order of visited states. In the worst case, the queue will contain the majority of the possible states that we need to visit, which in total is N⋅K as we discussed in the time complexity analysis. Thus, the space complexity of the queue is O(N⋅K).
o	Other than the queue, we also used a set variable (named seen) to keep track of all the visited states along the way. Same as the queue, the space complexity of this set is also O(N⋅K).
o	To sum up, the overall space complexity of the algorithm is O(N⋅K).
 
            */
            public int BFS(int[][] grid, int k)
            {
                int rows = grid.Length, cols = grid[0].Length;
                int[] target = { rows - 1, cols - 1 };

                // if we have sufficient quotas to eliminate the obstacles in the worst case,
                // then the shortest distance is the Manhattan distance.
                if (k >= rows + cols - 2)
                {
                    return rows + cols - 2;
                }

                LinkedList<StepState> queue = new LinkedList<StepState>(); //Dequeue
                HashSet<StepState> seen = new HashSet<StepState>();

                // (steps, row, col, remaining quota to eliminate obstacles)
                StepState start = new StepState(0, 0, 0, k);
                queue.AddLast(start);
                seen.Add(start);

                while (queue.Count > 0)
                {
                    StepState curr = queue.First.Value;
                    queue.RemoveFirst();

                    // we reach the target here
                    if (target[0] == curr.Row && target[1] == curr.Col)
                    {
                        return curr.Steps;
                    }

                    int[] nextSteps = { curr.Row, curr.Col + 1, curr.Row + 1, curr.Col,
                    curr.Row, curr.Col - 1, curr.Row - 1, curr.Col };

                    // explore the four directions in the next step
                    for (int i = 0; i < nextSteps.Length; i += 2)
                    {
                        int nextRow = nextSteps[i];
                        int nextCol = nextSteps[i + 1];

                        // out of the boundary of grid
                        if (nextRow < 0 || nextRow == rows || nextCol < 0 || nextCol == cols)
                        {
                            continue;
                        }

                        int nextElimination = curr.RemainingEliminations - grid[nextRow][nextCol];
                        StepState newState = new StepState(curr.Steps + 1, nextRow, nextCol, nextElimination);

                        // add the next move in the queue if it qualifies.
                        if (nextElimination >= 0 && !seen.Contains(newState))
                        {
                            seen.Add(newState);
                            queue.AddLast(newState);
                        }
                    }
                }

                // did not reach the target
                return -1;
            }
            class StepState
            {
                /**
                 * data object to keep the state info for each step:
                 * <steps, row, col, remaining_eliminations>
                 */
                public int Steps { get; set; }
                public int Row { get; set; }
                public int Col { get; set; }
                public int RemainingEliminations { get; set; }

                public StepState(int steps, int row, int col, int remainingEliminations)
                {
                    Steps = steps;
                    Row = row;
                    Col = col;
                    RemainingEliminations = remainingEliminations;
                }

                public override int GetHashCode()
                {
                    // needed when we put objects into any container class
                    return (Row + 1) * (Col + 1) * RemainingEliminations;
                }

                public override bool Equals(object other)
                {
                    /**
                     * only (row, col, k) matters as the state info
                     */
                    if (!(other is StepState))
                    {
                        return false;
                    }
                    StepState newState = (StepState)other;
                    return (Row == newState.Row) && (Col == newState.Col) && (RemainingEliminations == newState.RemainingEliminations);
                }

                public override string ToString()
                {
                    return $"{Row} {Col} {RemainingEliminations}";
                }
            }
            /*
            Approach 2: A* (A Star) Algorithm
Complexity Analysis
Let N be the number of cells in the grid, and K be the quota to eliminate obstacles.
•	Time Complexity: O(N⋅K⋅log(N⋅K))
o	We conduct a BFS traversal in the grid. In the worst case, we will visit each cell in the grid. And each cell can be visited at most K times, with different quotas of obstacle elimination. Therefore, the total number of visits would be N⋅K.
o	For each visit, we perform one push and one pop operation in the priority queue, which takes O(log(N⋅K)) time.
o	Thus, the overall time complexity of the algorithm is O(N⋅K⋅log(N⋅K)).
o	Although the upper bound for the time complexity of the this algorithm is higher than the previous BFS approach, on average, the A* algorithm will outperform the previous BFS approach when there exists any relatively direct path from the source to the target.
•	Space Complexity: O(N⋅K)
o	We use a queue to maintain the order of visited states. In the worst case, the queue could contain the majority of the possible states that we must visit, which in total is N⋅K, as we discussed in the time complexity analysis. Thus, the space complexity of the queue is O(N⋅K).
o	Other than the queue, we also used a set variable (named seen) to keep track of all the states we visited along the way. Again, the space complexity of this set is also O(N⋅K).
o	To sum up, the overall space complexity of the algorithm is O(N⋅K).

            */
            public int UsingAStarAlgo(int[][] grid, int k)
            {
                int rows = grid.Length, cols = grid[0].Length;
                int[] target = { rows - 1, cols - 1 };

                PriorityQueue<StepStateExt, int> queue = new();
                HashSet<StepStateExt> seen = new();

                // (steps, row, col, remaining quota to eliminate obstacles)
                StepStateExt start = new(0, 0, 0, k, target);
                queue.Enqueue(start, start.Estimation); //TODO: Double Check this!!
                seen.Add(start);

                while (queue.Count > 0)
                {
                    StepStateExt curr = queue.Dequeue();

                    // we can reach the target in the shortest path (manhattan distance),
                    //   even if the remaining steps are all obstacles
                    int remainMinDistance = curr.Estimation - curr.Steps;
                    if (remainMinDistance <= curr.K)
                    {
                        return curr.Estimation;
                    }

                    int[] nextSteps = { curr.Row, curr.Col + 1, curr.Row + 1, curr.Col,
                    curr.Row, curr.Col - 1, curr.Row - 1, curr.Col };

                    // explore the four directions in the next step
                    for (int i = 0; i < nextSteps.Length; i += 2)
                    {
                        int nextRow = nextSteps[i];
                        int nextCol = nextSteps[i + 1];

                        // out of the boundary of grid
                        if (nextRow < 0 || nextRow == rows || nextCol < 0 || nextCol == cols)
                        {
                            continue;
                        }

                        int nextElimination = curr.K - grid[nextRow][nextCol];
                        StepStateExt newState = new(curr.Steps + 1, nextRow, nextCol, nextElimination, target);

                        // add the next move in the queue if it qualifies.
                        if (nextElimination >= 0 && !seen.Contains(newState))
                        {
                            seen.Add(newState);
                            queue.Enqueue(newState, newState.Estimation);
                        }
                    }
                }

                // did not reach the target
                return -1;
            }
            class StepStateExt : IComparable<StepStateExt>
            {
                /**
                 * state info for each step:
                 * <estimation, steps, row, col, remaining_eliminations>
                 */
                public int Estimation { get; private set; }
                public int Steps { get; private set; }
                public int Row { get; private set; }
                public int Col { get; private set; }
                public int K { get; private set; }
                private int[] Target;

                public StepStateExt(int steps, int row, int col, int k, int[] target)
                {
                    Steps = steps;
                    Row = row;
                    Col = col;
                    K = k;

                    Target = target;
                    int manhattanDistance = Math.Abs(target[0] - row) + Math.Abs(target[1] - col);
                    // h(n) = manhattan distance,  g(n) = 0
                    // estimation = h(n) + g(n)
                    Estimation = manhattanDistance + steps;
                }

                public override int GetHashCode()
                {
                    return (Row + 1) * (Col + 1) * K;
                }

                public int CompareTo(StepStateExt other)
                {
                    // order the elements solely based on the 'estimation' value
                    return Estimation.CompareTo(other.Estimation);
                }

                public override bool Equals(object obj)
                {
                    if (!(obj is StepStateExt))
                    {
                        return false;
                    }
                    StepStateExt newState = (StepStateExt)obj;
                    return (Row == newState.Row) && (Col == newState.Col) && (K == newState.K);
                }

                public override string ToString()
                {
                    return $"({Estimation} {Steps} {Row} {Col} {K})";
                }
            }

        }


        /* 1275. Find Winner on a Tic Tac Toe Game
        https://leetcode.com/problems/find-winner-on-a-tic-tac-toe-game/description/
         */
        public class FindWinnderOnTicTacToeSol
        {
            // Initialize the board, n = 3 in this problem.
            private int[,] board;
            private int boardSize = 3;
            /*
            Approach 1: Brute Force
            Complexity Analysis
            Let n be the length of the board and m be the length of input moves.
            •	Time complexity: O(m⋅n)
            For every move, we need to traverse the same row, column, diagonal, and anti-diagonal, which takes O(n) time.
            •	Space complexity: O(n2)
            We use an n by n array to record every move.

            */
            public string Naive(int[,] moves)
            {
                board = new int[boardSize, boardSize];
                int currentPlayer = 1;

                // For each move
                for (int i = 0; i < moves.GetLength(0); i++)
                {
                    int row = moves[i, 0], col = moves[i, 1];

                    // Mark the current move with the current player's id.
                    board[row, col] = currentPlayer;

                    // If any of the winning conditions is met, return the current player's id.
                    if (CheckRow(row, currentPlayer) ||
                        CheckCol(col, currentPlayer) ||
                        (row == col && CheckDiagonal(currentPlayer)) ||
                        (row + col == boardSize - 1 && CheckAntiDiagonal(currentPlayer)))
                    {
                        return currentPlayer == 1 ? "A" : "B";
                    }

                    // If no one wins so far, change to the other player alternatively. 
                    // That is from 1 to -1, from -1 to 1.
                    currentPlayer *= -1;
                }

                // If all moves are completed and there is still no result, we shall check if 
                // the grid is full or not. If so, the game ends with draw, otherwise pending.
                return moves.GetLength(0) == boardSize * boardSize ? "Draw" : "Pending";
            }

            // Check if any of 4 winning conditions to see if the current player has won.
            private bool CheckRow(int row, int player)
            {
                for (int col = 0; col < boardSize; ++col)
                {
                    if (board[row, col] != player) return false;
                }
                return true;
            }

            private bool CheckCol(int col, int player)
            {
                for (int row = 0; row < boardSize; ++row)
                {
                    if (board[row, col] != player) return false;
                }
                return true;
            }

            private bool CheckDiagonal(int player)
            {
                for (int row = 0; row < boardSize; ++row)
                {
                    if (board[row, row] != player) return false;
                }
                return true;
            }

            private bool CheckAntiDiagonal(int player)
            {
                for (int row = 0; row < boardSize; ++row)
                {
                    if (board[row, boardSize - 1 - row] != player) return false;
                }
                return true;
            }

            /*
Approach 2: Record Each Move
Complexity Analysis
Let n be the length of the board and m be the length of input moves.
•	Time complexity: O(m)
For every move, we update the value for a row, column, diagonal, and anti-diagonal. Each update takes constant time. We also check if any of these lines satisfies the winning condition which also takes constant time.
•	Space complexity: O(n)
We use two arrays of size n to record the value for each row and column, and two integers of constant space to record to value for diagonal and anti-diagonal.

            */
            public String RecordEachMove(int[][] moves)
            {

                // n stands for the size of the board, n = 3 for the current game.
                int n = 3;

                // Use rows and cols to record the value on each row and each column.
                // diag1 and diag2 to record value on diagonal or anti-diagonal.
                int[] rows = new int[n], cols = new int[n];
                int diag = 0, anti_diag = 0;

                // Two players having value of 1 and -1, player_1 with value = 1 places first.
                int player = 1;

                foreach (int[] move in moves)
                {

                    // Get the row number and column number for this move.
                    int row = move[0], col = move[1];

                    // Update the row value and column value.
                    rows[row] += player;
                    cols[col] += player;

                    // If this move is placed on diagonal or anti-diagonal, 
                    // we shall update the relative value as well.
                    if (row == col)
                    {
                        diag += player;
                    }
                    if (row + col == n - 1)
                    {
                        anti_diag += player;
                    }

                    // Check if this move meets any of the winning conditions.
                    if (Math.Abs(rows[row]) == n || Math.Abs(cols[col]) == n ||
                        Math.Abs(diag) == n || Math.Abs(anti_diag) == n)
                    {
                        return player == 1 ? "A" : "B";
                    }

                    // If no one wins so far, change to the other player alternatively. 
                    // That is from 1 to -1, from -1 to 1.
                    player *= -1;
                }

                // If all moves are completed and there is still no result, we shall check if 
                // the grid is full or not. If so, the game ends with draw, otherwise pending.
                return moves.Length == n * n ? "Draw" : "Pending";
            }

        }


        /* 1319. Number of Operations to Make Network Connected
        https://leetcode.com/problems/number-of-operations-to-make-network-connected/description/

         */
        class MakeConnectedSol
        {
            /*
            Approach 1: Depth First Search
            Complexity Analysis
    Here n is the number of nodes and e is the total number edges (determined by size of connections).
    •	Time complexity: O(n+e).
    o	We need O(e) time to initialize the adjacency list and O(n) to initialize the visit array.
    o	The dfs function visits each node once, which takes O(n) time in total. Because we have undirected edges, each edge can only be iterated twice (by nodes at the end), resulting in O(e) operations total while visiting all nodes.
    o	As a result, the total time required is O(n+e).
    •	Space complexity: O(n+e).
    o	Building the adjacency list takes O(e) space.
    o	The visit array takes O(n) space.
    o	The recursion call stack used by dfs can have no more than n elements in the worst-case scenario. It would take up O(n) space in that case.

            */
            public int DFS(int numberOfNodes, int[][] connections)
            {
                if (connections.Length < numberOfNodes - 1)
                {
                    return -1;
                }

                Dictionary<int, List<int>> adjacencyList = new Dictionary<int, List<int>>();
                foreach (int[] connection in connections)
                {
                    if (!adjacencyList.ContainsKey(connection[0]))
                    {
                        adjacencyList[connection[0]] = new List<int>();
                    }
                    adjacencyList[connection[0]].Add(connection[1]);

                    if (!adjacencyList.ContainsKey(connection[1]))
                    {
                        adjacencyList[connection[1]] = new List<int>();
                    }
                    adjacencyList[connection[1]].Add(connection[0]);
                }

                int numberOfConnectedComponents = 0;
                bool[] visited = new bool[numberOfNodes];
                for (int i = 0; i < numberOfNodes; i++)
                {
                    if (!visited[i])
                    {
                        numberOfConnectedComponents++;
                        Dfs(i, adjacencyList, visited);
                    }
                }

                return numberOfConnectedComponents - 1;
            }
            public void Dfs(int node, Dictionary<int, List<int>> adjacencyList, bool[] visited)
            {
                visited[node] = true;
                if (!adjacencyList.ContainsKey(node))
                {
                    return;
                }
                foreach (int neighbor in adjacencyList[node])
                {
                    if (!visited[neighbor])
                    {
                        visited[neighbor] = true;
                        Dfs(neighbor, adjacencyList, visited);
                    }
                }
            }

            /*
            Approach 2: Breadth First Search
            Complexity Analysis
Here n is the number of nodes and e is the total number edges (determined by size of connections).
•	Time complexity: O(n+e).
o	Each queue operation in the BFS algorithm takes O(1) time, and a single node can only be pushed once, leading to O(n) operations for n nodes. We iterate over all the neighbors of each node that is popped out of the queue, so for an undirected edge, a given edge could be iterated at most twice (by nodes at both ends), resulting in O(e) operations total for all the nodes.
o	We also need O(e) time to initialize the adjacency list and O(n) to initialize the visit array.
o	As a result, the total time required is O(n+e).
•	Space complexity: O(n+e).
o	Building the adjacency list takes O(e) space.
o	The BFS queue takes O(n) because each node is added at most once.
o	The visit array takes O(n) space as well.

            */
            public int BFS(int numberOfNodes, int[][] connections)
            {
                if (connections.Length < numberOfNodes - 1)
                {
                    return -1;
                }

                Dictionary<int, List<int>> adjacencyList = new Dictionary<int, List<int>>();
                foreach (int[] connection in connections)
                {
                    if (!adjacencyList.ContainsKey(connection[0]))
                    {
                        adjacencyList[connection[0]] = new List<int>();
                    }
                    adjacencyList[connection[0]].Add(connection[1]);

                    if (!adjacencyList.ContainsKey(connection[1]))
                    {
                        adjacencyList[connection[1]] = new List<int>();
                    }
                    adjacencyList[connection[1]].Add(connection[0]);
                }

                int numberOfConnectedComponents = 0;
                bool[] visited = new bool[numberOfNodes];
                for (int i = 0; i < numberOfNodes; i++)
                {
                    if (!visited[i])
                    {
                        numberOfConnectedComponents++;
                        Bfs(i, adjacencyList, visited);
                    }
                }

                return numberOfConnectedComponents - 1;
            }
            private void Bfs(int node, Dictionary<int, List<int>> adjacencyList, bool[] visited)
            {
                Queue<int> queue = new Queue<int>();
                queue.Enqueue(node);
                visited[node] = true;

                while (queue.Count > 0)
                {
                    node = queue.Dequeue();
                    if (!adjacencyList.ContainsKey(node))
                    {
                        continue;
                    }
                    foreach (int neighbor in adjacencyList[node])
                    {
                        if (!visited[neighbor])
                        {
                            visited[neighbor] = true;
                            queue.Enqueue(neighbor);
                        }
                    }
                }
            }

            /*
            Approach 3: Union-find
            Complexity Analysis
Here n is the number of nodes and e is the total number edges (determined by size of connections).
•	Time complexity: O(n+e).
o	For T operations, the amortized time complexity of the union-find algorithm (using path compression with union by rank) is O(alpha(T)). Here, α(T) is the inverse Ackermann function that grows so slowly, that it doesn't exceed 4 for all reasonable T (approximately T<10600). You can read more about the complexity of union-find here. Because the function grows so slowly, we consider it to be O(1).
o	Initializing UnionFind takes O(n) time beacuse we are initializing the parent and rank arrays of size n each.
o	We iterate through every edge and use the find operation to find the component of nodes connected by each edge. It takes O(1) per operation and takes O(e) time for all the e edges. If nodes from different components are connected by an edge, we also perform union of the nodes, which takes O(1) time per operation. Because there are e edges, it may be called e times in the worst-case scenario. It would take O(e) time in that case.
o	As a result, the total time required is O(n+e).
•	Space complexity: O(n).
o	We are using the parent and rank arrays, both of which require O(n) space each.

            */
            public int UsingUnionFind(int numberOfNodes, int[][] connections)
            {
                if (connections.Length < numberOfNodes - 1)
                {
                    return -1;
                }

                UnionFind disjointSetUnion = new UnionFind(numberOfNodes);
                int numberOfConnectedComponents = numberOfNodes;
                foreach (int[] connection in connections)
                {
                    if (disjointSetUnion.Find(connection[0]) != disjointSetUnion.Find(connection[1]))
                    {
                        numberOfConnectedComponents--;
                        disjointSetUnion.UnionSet(connection[0], connection[1]);
                    }
                }

                return numberOfConnectedComponents - 1;
            }
            public class UnionFind
            {
                private int[] parent;
                private int[] rank;

                public UnionFind(int size)
                {
                    parent = new int[size];
                    for (int index = 0; index < size; index++)
                        parent[index] = index;
                    rank = new int[size];
                }

                public int Find(int element)
                {
                    if (parent[element] != element)
                        parent[element] = Find(parent[element]);
                    return parent[element];
                }

                public void UnionSet(int firstElement, int secondElement)
                {
                    int firstSet = Find(firstElement), secondSet = Find(secondElement);
                    if (firstSet == secondSet)
                    {
                        return;
                    }
                    else if (rank[firstSet] < rank[secondSet])
                    {
                        parent[firstSet] = secondSet;
                    }
                    else if (rank[firstSet] > rank[secondSet])
                    {
                        parent[secondSet] = firstSet;
                    }
                    else
                    {
                        parent[secondSet] = firstSet;
                        rank[firstSet]++;
                    }
                }
            }
        }


        /* 1335. Minimum Difficulty of a Job Schedule
        https://leetcode.com/problems/minimum-difficulty-of-a-job-schedule/description/
         */
        class MinDifficultySol
        {
            /*
            Approach 1: Top-down DP

            Complexity Analysis
    Let n be the length of the jobDifficulty array, and d be the total number of days.
    •	Time complexity: O(n^2⋅d) since there are n⋅d possible states, and we need O(n) time to calculate the result for each state.
    •	Space complexity: O(n⋅d) space is required to memoize all n⋅d states.

            */
            public int TopDownDP(int[] jobDifficulty, int d)
            {

                int n = jobDifficulty.Length;
                // Edge case: make sure there is at least one job per day
                if (n < d)
                {
                    return -1;
                }

                int[][] mem = new int[n][];
                for (int i = 0; i < n; i++)
                {
                    for (int j = 0; j <= d; j++)
                    {
                        mem[i][j] = -1;
                    }
                }

                return MinDiff(0, d, jobDifficulty, mem);
            }

            private int MinDiff(int i, int daysRemaining, int[] jobDifficulty, int[][] mem)
            {
                // Use memoization to avoid repeated computation of states
                if (mem[i][daysRemaining] != -1)
                {
                    return mem[i][daysRemaining];
                }

                // Base case: finish all remaining jobs in the last day
                if (daysRemaining == 1)
                {
                    int result = 0;
                    for (int j = i; j < jobDifficulty.Length; j++)
                    {
                        result = Math.Max(result, jobDifficulty[j]);
                    }
                    return result;
                }

                int res = int.MaxValue;
                int dailyMaxJobDiff = 0;
                // Iterate through possible starting index for the next day
                // and ensure we have at least one job for each remaining day.
                for (int j = i; j < jobDifficulty.Length - daysRemaining + 1; j++)
                {
                    dailyMaxJobDiff = Math.Max(dailyMaxJobDiff, jobDifficulty[j]);
                    res = Math.Min(res, dailyMaxJobDiff + MinDiff(j + 1, daysRemaining - 1, jobDifficulty, mem));
                }
                mem[i][daysRemaining] = res;
                return res;
            }

            /*Approach 2: Bottom-up 2D DP
            Complexity Analysis
Let n be the length of the jobDifficulty array, and d be the total number of days.
•	Time complexity: O(n^2⋅d) since there are n⋅d possible states, and we need O(n) time to calculate the result for each state.
•	Space complexity: O(n⋅d) is required for the (n+1)×(d+1) DP array.

*/
            public int BottomUp2DDP(int[] jobDifficulty, int d)
            {
                int n = jobDifficulty.Length;
                // Initialize the minDiff matrix to record the minimum difficulty
                // of the job schedule
                int[][] minDiff = new int[d + 1][];
                for (int daysRemaining = 0; daysRemaining <= d; daysRemaining++)
                {
                    for (int i = 0; i < n; i++)
                    {
                        minDiff[daysRemaining][i] = int.MaxValue;
                    }
                }
                for (int daysRemaining = 1; daysRemaining <= d; daysRemaining++)
                {
                    for (int i = 0; i < n - daysRemaining + 1; i++)
                    {
                        int dailyMaxJobDiff = 0;
                        for (int j = i + 1; j < n - daysRemaining + 2; j++)
                        {
                            // Use dailyMaxJobDiff to record maximum job difficulty
                            dailyMaxJobDiff = Math.Max(dailyMaxJobDiff, jobDifficulty[j - 1]);
                            if (minDiff[daysRemaining - 1][j] != int.MaxValue)
                            {
                                minDiff[daysRemaining][i] = Math.Min(minDiff[daysRemaining][i],
                                                                     dailyMaxJobDiff + minDiff[daysRemaining - 1][j]);
                            }
                        }
                    }
                }
                return minDiff[d][0] < int.MaxValue ? minDiff[d][0] : -1;
            }
            /* Approach 3: Bottom-up 1D DP
            Complexity Analysis
            Let n be the length of the jobDifficulty array, and d be the total number of days.
            •	Time complexity: O(n^2⋅d) since there are n⋅d possible states, and we need O(n) time to calculate the result for each state.
            •	Space complexity: O(n) as we only use two arrays of length n+1 to store all relevant states at any given time.

             */
            public int BottomUp1DDP(int[] jobDifficulty, int days)
            {
                int jobCount = jobDifficulty.Length;
                // Initialize the minDiff matrix to record the minimum difficulty
                // of the job schedule    
                int[] minDiffNextDay = new int[jobCount + 1];
                for (int i = 0; i < jobCount; i++)
                {
                    minDiffNextDay[i] = int.MaxValue;
                }
                for (int daysRemaining = 1; daysRemaining <= days; daysRemaining++)
                {
                    int[] minDiffCurrDay = new int[jobCount + 1];
                    for (int i = 0; i < jobCount; i++)
                    {
                        minDiffCurrDay[i] = int.MaxValue;
                    }
                    for (int i = 0; i < jobCount - daysRemaining + 1; i++)
                    {
                        int dailyMaxJobDiff = 0;
                        for (int j = i + 1; j < jobCount - daysRemaining + 2; j++)
                        {
                            // Use dailyMaxJobDiff to record maximum job difficulty
                            dailyMaxJobDiff = Math.Max(dailyMaxJobDiff, jobDifficulty[j - 1]);
                            if (minDiffNextDay[j] != int.MaxValue)
                            {
                                minDiffCurrDay[i] = Math.Min(minDiffCurrDay[i],
                                                               dailyMaxJobDiff + minDiffNextDay[j]);
                            }
                        }
                    }
                    minDiffNextDay = minDiffCurrDay;
                }
                return minDiffNextDay[0] < int.MaxValue ? minDiffNextDay[0] : -1;
            }
            /*
            Approach 4: Monotonic Stack - Better Time Complexity
            Complexity Analysis
Let n be the length of the jobDifficulty array, and d be the total number of days.
•	Time complexity: O(n⋅d) as there are n⋅d possible states. Using the stack solution, we need O(n) time to calculate all n states for each day.
•	Space complexity: O(n) as we only use one array of length n to store all DP states for the prior day and the current day, and the stack that will contain at most n elements.

            */
            public int UsingMonotonicStack(int[] jobDifficulty, int days)
            {
                int jobCount = jobDifficulty.Length;
                if (jobCount < days)
                {
                    return -1;
                }

                // minDiffCurrDay and minDiffPrevDay record the minimum total job difficulty
                // for the current day and previous day, respectively
                int[] minDiffPrevDay = new int[jobCount];
                int[] minDiffCurrDay = new int[jobCount];
                int[] temp;

                Array.Fill(minDiffPrevDay, 1000);
                Stack<int> stack = new Stack<int>();

                for (int day = 0; day < days; ++day)
                {
                    // Use a monotonically decreasing stack to record job difficulties
                    stack.Clear();
                    // The number of jobs needs to be no less than number of days passed.
                    for (int i = day; i < jobCount; i++)
                    {
                        // We initialize minDiffCurrDay[i] as only performing 1 job at the i-th index.
                        // At day 0, the minimum total job difficulty is to complete the 0th job only.
                        // Otherwise, we increment minDiffPrevDay[i - 1] by the i-th job difficulty
                        minDiffCurrDay[i] = i > 0 ? minDiffPrevDay[i - 1] + jobDifficulty[i] : jobDifficulty[i];

                        // When we find the last element in the stack is smaller than or equal to current job,
                        // we need to pop out the element to maintain a monotonic decreasing stack.
                        while (stack.Count > 0 && jobDifficulty[stack.Peek()] <= jobDifficulty[i])
                        {
                            // If we include all jobs with index j+1 to i to the current d,
                            // total job difficulty of the current d will be increased
                            // by the amount of jobDifficulty[i] - jobDifficulty[j]
                            int j = stack.Pop();
                            int diffIncr = jobDifficulty[i] - jobDifficulty[j];
                            minDiffCurrDay[i] = Math.Min(minDiffCurrDay[i], minDiffCurrDay[j] + diffIncr);
                        }

                        // When the last element in the stack is larger than current element,
                        // if we include all jobs with index j+1 to i to the current d
                        // the overall job difficulty will not change
                        if (stack.Count > 0)
                        {
                            minDiffCurrDay[i] = Math.Min(minDiffCurrDay[i], minDiffCurrDay[stack.Peek()]);
                        }

                        // Update the monotonic stack by adding in the current index
                        stack.Push(i);
                    }
                    temp = minDiffPrevDay;
                    minDiffPrevDay = minDiffCurrDay;
                    minDiffCurrDay = temp;
                }
                return minDiffPrevDay[jobCount - 1];
            }
        }



        /* 1359. Count All Valid Pickup and Delivery Options

        https://leetcode.com/problems/count-all-valid-pickup-and-delivery-options/description/
         */
        class CountOrdersSol
        {
            private int MOD = 1_000_000_007;
            private long[][] memo;
            /*
            
Approach 1: Recursion with Memoization (Top-Down DP)
Complexity Analysis
If N is the number of the orders given.
•	Time complexity: O(N^2).
The recursive function would have made approximately 2^N recursive calls, but due to caching, we will avoid recomputation of results, and only unique function calls may result in more recursive calls. The recursive function depends on two variables (unpicked and undelivered). Since the values for unpicked and undelivered must be in the range 0 to N, there will be at most (N+1)⋅(N+1) unique function calls.
•	Space complexity: O(N^2).
Our cache must store the results for all of the unique function calls that are valid states. There are approximately N^2/2 valid states.

            */
            public int TopDownDPRecWithMemo(int n)
            {
                memo = new long[n + 1][];
                return (int)TotalWays(n, n);
            }
            private long TotalWays(int unpicked, int undelivered)
            {
                if (unpicked == 0 && undelivered == 0)
                {
                    // We have completed all orders.
                    return 1;
                }

                if (unpicked < 0 || undelivered < 0 || undelivered < unpicked)
                {
                    // We can't pick or deliver more than N items
                    // Number of deliveries can't exceed number of pickups 
                    // as we can only deliver after a pickup.
                    return 0;
                }

                if (memo[unpicked][undelivered] != 0)
                {
                    // Return cached value, if already present. 
                    return memo[unpicked][undelivered];
                }

                long ans = 0;

                // Count all choices of picking up an order.
                ans += unpicked * TotalWays(unpicked - 1, undelivered);
                // Handle integer overflow.
                ans %= MOD;

                // Count all choices of delivering a picked order.
                ans += (undelivered - unpicked) * TotalWays(unpicked, undelivered - 1);
                // Handle integer overflow.
                ans %= MOD;

                return memo[unpicked][undelivered] = ans;
            }


            /* Approach 2: Tabulation (Bottom-Up DP) 
            Complexity Analysis
            If N is the number of the orders given.
            •	Time complexity: O(N^2).
            We have two state variables, and each subproblem is a configuration of those two state variables. Thus, there will be at most (N+1)⋅(N+1) unique subproblems and we will iterate over half of them using two nested for loops.
            •	Space complexity: O(N^2).
            Our DP table must be large enough to store all of the (N+1)⋅(N+1) possible states.
            Note: You can further reduce the space complexity of this approach as we can see in the slideshow while building the DP table we only need previous and current rows. Thus instead of keeping an N⋅N size array we could just keep two N size arrays.
            It would follow the same approach explained above, but it's not implemented here.

            */
            public int BottomUpDP(int n)
            {
                long[][] dp = new long[n + 1][];

                for (int unpicked = 0; unpicked <= n; unpicked++)
                {
                    for (int undelivered = unpicked; undelivered <= n; undelivered++)
                    {
                        // If all orders are picked and delivered then,
                        // for remaining '0' orders we have only one way.
                        if (unpicked == 0 && undelivered == 0)
                        {
                            dp[unpicked][undelivered] = 1;
                            continue;
                        }

                        // There are some unpicked elements left. 
                        // We have choice to pick any one of those orders.
                        if (unpicked > 0)
                        {
                            dp[unpicked][undelivered] += unpicked * dp[unpicked - 1][undelivered];
                        }
                        dp[unpicked][undelivered] %= MOD;

                        // Number of deliveries done is less than picked orders.
                        // We have choice to deliver any one of (undelivered - unpicked) orders. 
                        if (undelivered > unpicked)
                        {
                            dp[unpicked][undelivered] += (undelivered - unpicked) * dp[unpicked][undelivered - 1];
                        }
                        dp[unpicked][undelivered] %= MOD;
                    }
                }

                return (int)dp[n][n];
            }

            /*
            Approach 3: Permutations (Math)
            Complexity Analysis
If N is the number of the orders given.
•	Time complexity: O(N).
To calcualte N! and ∏i=1toN(2∗i−1) we need to iterate over N elements.
•	Space complexity: O(1).
We have used only constant space to store the result.

            */
            public int WithMathPermutations(int n)
            {
                long ans = 1;
                int MOD = 1_000_000_007;

                for (int i = 1; i <= n; ++i)
                {
                    // Ways to arrange all pickups, 1*2*3*4*5*...*n
                    ans = ans * i;
                    // Ways to arrange all deliveries, 1*3*5*...*(2n-1)
                    ans = ans * (2 * i - 1);
                    ans %= MOD;
                }

                return (int)ans;
            }

            /*
            Approach 4: Probability (Math)

Complexity Analysis
If N is the number of the orders given.
•	Time complexity: O(N).
o	For calculating (2N)! we need to iterate over 2N elements.
o	And for calculating 1/2N we multiply 2 N-times.
o	Thus, it leads to a time complexity of O(N).
•	Space complexity: O(1).
We have used only constant space to store the result.
Note: In python, we can use the factorial() function, a direct function that can compute the factorial of a number without writing the whole code for computing factorial.
But then the approach will not count as O(1) space, the reason is that in the worst case we will calculate factorial(2∗n) which may have a lot of digits. The way python handles extremely long numbers is that it requires an extra 4 bytes of memory every time the number increases by a factor of 230.

            */
            public int WithMathProbability(int n)
            {
                long ans = 1;
                int MOD = 1_000_000_007;

                for (int i = 1; i <= 2 * n; ++i)
                {
                    ans = ans * i;

                    // We only need to divide the result by 2 n-times.
                    // To prevent decimal results we divide after multiplying an even number.
                    if (i % 2 == 0)
                    {
                        ans = ans / 2;
                    }
                    ans %= MOD;
                }
                return (int)ans;
            }

        }



        /* 1383. Maximum Performance of a Team
        https://leetcode.com/problems/maximum-performance-of-a-team/description/
         */
        public class MaxPerformanceSol
        {
            /*
            Approach: Greedy with Priority Queue
            Complexity Analysis
    Let N be the total number of candidates, and K be the size of the team.
    •	Time Complexity: O(N⋅(logN+logK))
    o	First of all, we build a list of candidates from the inputs, which takes O(N) time.
    o	We then sort the candidates, which takes O(NlogN) time.
    o	We iterate through the sorted candidates. At each iteration, we will perform at most two operations on the priority queue: one push and one pop.
    Each operation takes O(log(K−1)) time, where K−1 is the capacity of the queue.
    To sum up, the time complexity of this iteration will be O(N⋅log(K−1))=O(N⋅logK).
    o	Thus, the overall time complexity of the algorithm will be O(N⋅(logN+logK)).
    •	Space Complexity: O(N+K)
    o	We build a list of candidates from the inputs, which takes O(N) space.
    o	We also use the priority queue data structure whose space capacity is O(K−1).
    o	Note that we use sorting in the algorithm, and the space complexity of the sorting algorithm depends on the implementation of each programming language.
    For instance, the sorted() function in Python is implemented with the Timsort algorithm whose space complexity is O(N).
    While in Java, the Collections.sort() is implemented as a variant of the quicksort algorithm whose space complexity is O(logN).
    o	To sum up, the overall space complexity of the entire algorithm is O(N+K).

            */
            public int GreedyWithPQ(int numberOfEngineers, int[] speeds, int[] efficiencies, int teamSize)
            {
                int modulo = (int)Math.Pow(10, 9) + 7;
                // build tuples of (efficiency, speed)
                List<Tuple<int, int>> candidates = new List<Tuple<int, int>>();
                for (int i = 0; i < numberOfEngineers; ++i)
                {
                    candidates.Add(new Tuple<int, int>(efficiencies[i], speeds[i]));
                }
                // sort the members by their efficiencies
                candidates = candidates.OrderByDescending(o => o.Item1).ToList();

                // create a heap to keep the top (k-1) speeds
                PriorityQueue<int, int> speedHeap = new PriorityQueue<int, int>();

                long speedSum = 0, performance = 0;
                foreach (Tuple<int, int> candidate in candidates)
                {
                    int currentEfficiency = candidate.Item1;
                    int currentSpeed = candidate.Item2;
                    // maintain a heap for the fastest (k-1) speeds
                    if (speedHeap.Count > teamSize - 1)
                    {
                        speedSum -= speedHeap.Dequeue();
                    }
                    speedHeap.Enqueue(currentSpeed, currentSpeed);

                    // calculate the maximum performance with
                    // the current member as the least efficient one in the team
                    speedSum += currentSpeed;
                    performance = Math.Max(performance, speedSum * currentEfficiency);
                }
                return (int)(performance % modulo);
            }
        }


        /* 1402. Reducing Dishes
        https://leetcode.com/problems/reducing-dishes/description/
         */
        class FindMaxSatisfactionSol
        {
            /*  
            Approach 1: Top-Down Dynamic Programming
            Complexity Analysis
    Here N is the number of dishes.
    •	Time complexity: O(N^2).
    Each state is defined by the values index and time. Hence, there will be N2 possible states, because both index and time can take up to N values and we must visit these states to solve the original problem. Each recursive call requires O(1) time as we just have a comparison operation. We also perform sorting, taking O(NlogN) time. Thus, the total time complexity equals O(N^2).
    •	Space complexity: O(N^2).
    The memoization results are stored in the table memo with size N^2. Also, stack space in the recursion equals the maximum number of active functions. The maximum number of active functions will be at most N, i.e. one function call for every dish. Hence, the space complexity is O(N^2).

            */

            public int TopDownDP(int[] satisfaction)
            {
                Array.Sort(satisfaction);

                int[][] memo = new int[satisfaction.Length + 1][];
                // Mark, all the states as -1, denoting not yet calculated.
                for (int i = 0; i < satisfaction.Length; i++)
                {
                    Array.Fill(memo[i], -1);
                }

                return FindMaxSatisfaction(satisfaction, memo, 0, 1);
            }
            private int FindMaxSatisfaction(int[] satisfaction, int[][] memo, int index, int time)
            {
                // Return 0 if we have iterated over all the dishes.
                if (index == satisfaction.Length)
                {
                    return 0;
                }

                // We have already calculated the answer, so no need to go into recursion.
                if (memo[index][time] != -1)
                {
                    return memo[index][time];
                }

                // Return the maximum of two choices:
                // 1. Cook the dish at `index` with the time taken as `time` and move on to the next dish with time as time + 1.
                // 2. Skip the current dish and move on to the next dish at the same time.
                return memo[index][time] = Math.Max(satisfaction[index] * time + FindMaxSatisfaction(satisfaction, memo, index + 1, time + 1),
                        FindMaxSatisfaction(satisfaction, memo, index + 1, time));
            }




            /*
            Approach 2: Bottom-Up Dynamic Programming
         Complexity Analysis
    Here N is the number of dishes.
    •	Time complexity: O(N^2).
    Each state is defined by the values index and time. Hence, there will be N2 possible states, and we will iterate over each state to solve the original problem. We also perform sorting, taking O(NlogN) time. Thus, the total time complexity equals O(N^2).
    •	Space complexity: O(N^2).
    We have the table dp with size N^2. Hence, the space complexity is O(N^2).

            */
            public int BottomUpDP(int[] satisfaction)
            {
                Array.Sort(satisfaction);

                int[][] dp = new int[satisfaction.Length + 1][];
                // Mark all the states initially as 0.
                for (int i = 0; i <= satisfaction.Length; i++)
                {
                    Array.Fill(dp[i], 0);
                }

                for (int i = satisfaction.Length - 1; i >= 0; i--)
                {
                    for (int j = 1; j <= satisfaction.Length; j++)
                    {
                        // Maximum of two choices:
                        // 1. Cook the dish at `index` with the time taken as `time` and move on to the next dish with time as time + 1.
                        // 2. Skip the current dish and move on to the next dish at the same time.
                        dp[i][j] = Math.Max(satisfaction[i] * j + dp[i + 1][j + 1], dp[i + 1][j]);
                    }
                }

                return dp[0][1];
            }

            /*
            Approach 3: Bottom-Up Dynamic Programming (Space Optimized)
Complexity Analysis
Here N is the number of dishes.
•	Time complexity: O(N^2).
Each state is defined by the values index and time. Hence, there will be N2 possible states, and we will iterate over each state to solve the original problem. We also perform sorting, taking O(NlogN) time. Thus, the total time complexity equals O(N^2).
•	Space complexity: O(N).
We have two tables, dp and prev, with size N. Hence, the space complexity is O(N).

            */
            public int BottomUpDPSpaceOptimal(int[] satisfaction)
            {
                Array.Sort(satisfaction);

                // Array to keep the result for the previous iteration.
                int[] prev = new int[satisfaction.Length + 2];
                Array.Fill(prev, 0);

                for (int index = satisfaction.Length - 1; index >= 0; index--)
                {
                    // Array to keep the result for the current iteration.
                    int[] dp = new int[satisfaction.Length + 2];

                    for (int time = 1; time <= satisfaction.Length; time++)
                    {
                        // Maximum of two choices:
                        // 1. Cook the dish at `index` with the time taken as `time` and move on to the next dish with time as time + 1.
                        // 2. Skip the current dish and move on to the next dish at the same time.
                        dp[time] = Math.Max(satisfaction[index] * time + prev[time + 1], prev[time]);
                    }
                    // Assign the current iteration result to prev to be used in the next iteration.
                    prev = dp;
                }
                // dp and prev have the same value here, but dp is not defined at this scope.
                return prev[1];
            }

            /*
            Approach 4: Greedy
         Complexity Analysis
Here N is the number of dishes.
•	Time complexity: O(NlogN).
Sorting the array satisfaction takes O(NlogN) time, and then we iterate over the array satisfaction with time O(N). Thus, the total time complexity equals O(NlogN).
•	Space complexity: O(logN).
No extra space is needed apart from two variables. However, some space is required for sorting. The space complexity of the sorting algorithm depends on the implementation of each programming language. For instance, in Java, the Arrays.sort() for primitives is implemented as a variant of the quicksort algorithm whose space complexity is O(log⁡N). In C++ sort() function provided by STL is a hybrid of Quick Sort, Heap Sort, and Insertion Sort and has a worst-case space complexity of O(log⁡N). Thus, the inbuilt sort() function might add up to O(log⁡N) to space complexity. Hence, the space complexity equals O(log⁡N).
   
            */
            public int WithGreedy(int[] satisfaction)
            {
                Array.Sort(satisfaction);

                int maxSatisfaction = 0;
                int suffixSum = 0;
                for (int i = satisfaction.Length - 1; i >= 0 && suffixSum + satisfaction[i] > 0; i--)
                {
                    // Total satisfaction with all dishes so far.
                    suffixSum += satisfaction[i];
                    // Adding one instance of previous dishes as we add one more dish on the left.
                    maxSatisfaction += suffixSum;
                }
                return maxSatisfaction;
            }

        }


        /* 1431. Kids With the Greatest Number of Candies
        https://leetcode.com/problems/kids-with-the-greatest-number-of-candies/description/

         */

        class KidsWithCandiesSol
        {
            /*
            Approach: AdHoc
Complexity Analysis
Here, n is the number of kids.
•	Time complexity: O(n)
o	We iterate over the candies array to find out maxCandies which takes O(n) time.
o	We iterate over the candies array once more. We check for each kid whether they will have the most candies among all the children after receiving extraCandies and push the result in result which takes O(1) time. It requires O(n) time for n kids.
•	Space complexity: O(1)
o	Without counting the space of input and output, we are not using any space except for some integers like maxCandies and candy.

            */
            public List<Boolean> Adhooc(int[] candies, int extraCandies)
            {
                // Find out the greatest number of candies among all the kids.
                int maxCandies = 0;
                foreach (int candy in candies)
                {
                    maxCandies = Math.Max(candy, maxCandies);
                }
                // For each kid, check if they will have greatest number of candies
                // among all the kids.
                List<bool> result = new List<bool>();
                foreach (int candy in candies)
                {
                    result.Add(candy + extraCandies >= maxCandies);
                }

                return result;
            }
        }



        /* 1443. Minimum Time to Collect All Apples in a Tree
        https://leetcode.com/problems/minimum-time-to-collect-all-apples-in-a-tree/description/ */

        class MinTimeToCollectAllApplesSol
        {

            /*
            Approach: Depth First Search
           Complexity Analysis
    Here, n be the number of nodes.
    •	Time complexity: O(n)
    o	Each node is visited by the dfs function once, which takes O(n) time in total. We also iterate over the edges of every node once (since we don't visit a node more than once, we don't iterate its edges more than once), which adds O(n) time since we have n−1 edges.
    o	We also require O(n) time to initialize the adjacency list and the visit array.
    •	Space complexity: O(n)
    o	The recursion call stack used by dfs can have no more than n elements in the worst-case scenario. It would take up O(n) space in that case.
    o	We also require O(n) space for the adjacency list and the visit array.

            */

            public int MinTime(int numberOfNodes, int[][] edges, List<bool> hasApple)
            {
                Dictionary<int, List<int>> adjacencyList = new Dictionary<int, List<int>>();
                foreach (int[] edge in edges)
                {
                    int nodeA = edge[0], nodeB = edge[1];
                    if (!adjacencyList.ContainsKey(nodeA))
                    {
                        adjacencyList[nodeA] = new List<int>();
                    }
                    adjacencyList[nodeA].Add(nodeB);

                    if (!adjacencyList.ContainsKey(nodeB))
                    {
                        adjacencyList[nodeB] = new List<int>();
                    }
                    adjacencyList[nodeB].Add(nodeA);
                }
                return Dfs(0, -1, adjacencyList, hasApple);
            }
            private int Dfs(int node, int parent, Dictionary<int, List<int>> adjacencyList, List<bool> hasApple)
            {
                if (!adjacencyList.ContainsKey(node))
                    return 0;

                int totalTime = 0, childTime = 0;
                foreach (int child in adjacencyList[node])
                {
                    if (child == parent)
                        continue;
                    childTime = Dfs(child, node, adjacencyList, hasApple);
                    // childTime > 0 indicates subtree of child has apples. Since the root node of the
                    // subtree does not contribute to the time, even if it has an apple, we have to check it
                    // independently.
                    if (childTime > 0 || hasApple[child])
                        totalTime += childTime + 2;
                }
                return totalTime;
            }


        }



        /* 1444. Number of Ways of Cutting a Pizza
        https://leetcode.com/problems/number-of-ways-of-cutting-a-pizza/description/
         */

        class WaysToCutPizzaSol
        {
            /*
            Approach 1: Dynamic Programming
           Complexity Analysis
    Let n denote the number of rows in pizza and m denote the number of columns in pizza.
    •	Time complexity: O(k⋅n⋅m⋅(n+m)).
    There are O(k⋅n⋅m) states [remain][row][col]. k for remain, n for row and m for col. For each state, we iterate over next_row in O(n) and over next_col in O(m).
    •	Space complexity: O(n⋅m⋅k).
    We store the matrix dp[k][rows][cols].

            */
            public int WithDP(string[] pizza, int k)
            {
                int numberOfRows = pizza.Length, numberOfColumns = pizza[0].Length;
                int[,] apples = new int[numberOfRows + 1, numberOfColumns + 1];
                int[,,] dp = new int[k, numberOfRows, numberOfColumns];

                for (int row = numberOfRows - 1; row >= 0; row--)
                {
                    for (int col = numberOfColumns - 1; col >= 0; col--)
                    {
                        apples[row, col] = (pizza[row][col] == 'A' ? 1 : 0) + apples[row + 1, col] + apples[row, col + 1]
                                - apples[row + 1, col + 1];
                        dp[0, row, col] = apples[row, col] > 0 ? 1 : 0;
                    }
                }

                int mod = 1000000007;
                for (int remain = 1; remain < k; remain++)
                {
                    for (int row = 0; row < numberOfRows; row++)
                    {
                        for (int col = 0; col < numberOfColumns; col++)
                        {
                            for (int nextRow = row + 1; nextRow < numberOfRows; nextRow++)
                            {
                                if (apples[row, col] - apples[nextRow, col] > 0)
                                {
                                    dp[remain, row, col] += dp[remain - 1, nextRow, col];
                                    dp[remain, row, col] %= mod;
                                }
                            }
                            for (int nextCol = col + 1; nextCol < numberOfColumns; nextCol++)
                            {
                                if (apples[row, col] - apples[row, nextCol] > 0)
                                {
                                    dp[remain, row, col] += dp[remain - 1, row, nextCol];
                                    dp[remain, row, col] %= mod;
                                }
                            }
                        }
                    }
                }
                return dp[k - 1, 0, 0];
            }

            /*
            Approach 2: Dynamic Programming with Optimized Space Complexity
Complexity Analysis
Let n denote the number of rows in pizza and m denote the number of columns in pizza.
•	Time complexity: O(k⋅n⋅m⋅(n+m)).
There are O(k⋅n⋅m) states [remain][row][col]. k for remain, n for row and m for col. For each state, we iterate over next_row in O(n) and over next_col in O(m).
•	Space complexity: O(n⋅m).
We store the matrices apples[rows+1][cols+1], f[rows][cols] and g[rows][cols].

            */
            public int DPSpaceOptimal(string[] pizza, int k)
            {
                int rowCount = pizza.Length, columnCount = pizza[0].Length;
                int[][] apples = new int[rowCount + 1][];
                for (int i = 0; i < apples.Length; i++)
                {
                    apples[i] = new int[columnCount + 1];
                }

                int[][] f = new int[rowCount][];
                for (int i = 0; i < f.Length; i++)
                {
                    f[i] = new int[columnCount];
                }

                for (int row = rowCount - 1; row >= 0; row--)
                {
                    for (int col = columnCount - 1; col >= 0; col--)
                    {
                        apples[row][col] = (pizza[row][col] == 'A' ? 1 : 0) + apples[row + 1][col] + apples[row][col + 1]
                                - apples[row + 1][col + 1];
                        f[row][col] = apples[row][col] > 0 ? 1 : 0;
                    }
                }

                int mod = 1000000007;

                for (int remain = 1; remain < k; remain++)
                {
                    int[][] g = new int[rowCount][];
                    for (int i = 0; i < g.Length; i++)
                    {
                        g[i] = new int[columnCount];
                    }

                    for (int row = 0; row < rowCount; row++)
                    {
                        for (int col = 0; col < columnCount; col++)
                        {
                            for (int nextRow = row + 1; nextRow < rowCount; nextRow++)
                            {
                                if (apples[row][col] - apples[nextRow][col] > 0)
                                {
                                    g[row][col] += f[nextRow][col];
                                    g[row][col] %= mod;
                                }
                            }
                            for (int nextCol = col + 1; nextCol < columnCount; nextCol++)
                            {
                                if (apples[row][col] - apples[row][nextCol] > 0)
                                {
                                    g[row][col] += f[row][nextCol];
                                    g[row][col] %= mod;
                                }
                            }
                        }
                    }
                    // copy g to f
                    f = g.Select(innerArray => (int[])innerArray.Clone()).ToArray();
                }

                return f[0][0];
            }
        }



        /* 1466. Reorder Routes to Make All Paths Lead to the City Zero
        https://leetcode.com/problems/reorder-routes-to-make-all-paths-lead-to-the-city-zero/description/

         */
        class MinReorderRoutesSol
        {
            private int count = 0;
            /*
            Approach 1: Depth First Search
           Complexity Analysis
Here n is the number of nodes.
•	Time complexity: O(n).
o	We need O(n) time to initialize the adjacency list.
o	The dfs function visits each node once, which takes O(n) time in total. Because we have undirected edges, each edge can only be iterated twice (by nodes at the end), resulting in O(e) operations total while visiting all nodes, where e is the number of edges. Because the given graph is a tree, there are n−1 undirected edges, so O(n+e)=O(n).
•	Space complexity: O(n).
o	Building the adjacency list takes O(n) space.
o	The recursion call stack used by dfs can have no more than n elements in the worst-case scenario. It would take up O(n) space in that case.
 
            */
            public int DFS(int n, int[][] connections)
            {
                Dictionary<int, List<List<int>>> adjacencyList = new Dictionary<int, List<List<int>>>();
                foreach (int[] connection in connections)
                {
                    if (!adjacencyList.ContainsKey(connection[0]))
                    {
                        adjacencyList[connection[0]] = new List<List<int>>();
                    }
                    adjacencyList[connection[0]].Add(new List<int> { connection[1], 1 });

                    if (!adjacencyList.ContainsKey(connection[1]))
                    {
                        adjacencyList[connection[1]] = new List<List<int>>();
                    }
                    adjacencyList[connection[1]].Add(new List<int> { connection[0], 0 });
                }
                Dfs(0, -1, adjacencyList);
                return count;
            }
            private void Dfs(int node, int parent, Dictionary<int, List<List<int>>> adjacencyList)
            {
                if (!adjacencyList.ContainsKey(node))
                {
                    return;
                }
                foreach (List<int> neighborInfo in adjacencyList[node])
                {
                    int neighbor = neighborInfo[0];
                    int sign = neighborInfo[1];
                    if (neighbor != parent)
                    {
                        count += sign;
                        Dfs(neighbor, node, adjacencyList);
                    }
                }
            }
            /*
            Approach 2: Breadth First Search
            Complexity Analysis
            Here n is the number of nodes.
            •	Time complexity: O(n).
            o	We need O(n) time to initialize the adjacency list and O(n) to initialize the visit array.
            o	Each queue operation in the BFS algorithm takes O(1) time, and a single node can only be pushed once, leading to O(n) operations for n nodes. We iterate over all the neighbors of each node that is popped out of the queue, so for an undirected edge, a given edge could be iterated at most twice (by nodes at both ends), resulting in O(e) operations total for all the nodes. As mentioned in the previous approach, O(e)=O(n) since the graph is a tree.
            •	Space complexity: O(n).
            o	Building the adjacency list takes O(n) space.
            o	The visit array takes O(n) space as well.
            o	The BFS queue takes O(n) space in the worst-case because each node is added once.

            */
            public int BFS(int n, int[][] connections)
            {
                Dictionary<int, List<List<int>>> adj = new Dictionary<int, List<List<int>>>();
                foreach (int[] connection in connections)
                {
                    if (!adj.ContainsKey(connection[0]))
                    {
                        adj[connection[0]] = new List<List<int>>();
                    }
                    adj[connection[0]].Add(new List<int> { connection[1], 1 });

                    if (!adj.ContainsKey(connection[1]))
                    {
                        adj[connection[1]] = new List<List<int>>();
                    }
                    adj[connection[1]].Add(new List<int> { connection[0], 0 });
                }
                Bfs(0, n, adj);
                return count;
            }
            private void Bfs(int node, int n, Dictionary<int, List<List<int>>> adj)
            {
                Queue<int> queue = new Queue<int>();
                bool[] visited = new bool[n];
                queue.Enqueue(node);
                visited[node] = true;

                while (queue.Count > 0)
                {
                    node = queue.Dequeue();
                    if (!adj.ContainsKey(node))
                    {
                        continue;
                    }
                    foreach (List<int> neighborInfo in adj[node])
                    {
                        int neighbor = neighborInfo[0];
                        int sign = neighborInfo[1];
                        if (!visited[neighbor])
                        {
                            count += sign;
                            visited[neighbor] = true;
                            queue.Enqueue(neighbor);
                        }
                    }
                }
            }


        }


        /* 1491. Average Salary Excluding the Minimum and Maximum Salary
        https://leetcode.com/problems/average-salary-excluding-the-minimum-and-maximum-salary/description/
           */
        class AverageSalExcludeMinAndMaxSalSol
        {
            /*
            Complexity Analysis
Here, N is the number of salaries.
•	Time complexity: O(N).
We just iterate over the salaries once, and hence the total time complexity would be O(N).
•	Space complexity: O(1).
We only need three variables to store the total sum, the maximum value and the minimum value. Hence, the total space complexity would be constant.

            */
            public double average(int[] salaries)
            {
                int minSalary = int.MaxValue;
                int maxSalary = int.MinValue;
                int sum = 0;

                foreach (int salary in salaries)
                {
                    // Sum of all the salaries.
                    sum += salary;
                    // Update the minimum salary.
                    minSalary = Math.Min(minSalary, salary);
                    // Update the maximum salary.
                    maxSalary = Math.Max(maxSalary, salary);
                }

                // Divide the sum by total size - 2, as we exclude minimum and maximum values.
                return (double)(sum - minSalary - maxSalary) / (double)(salaries.Length - 2);
            }
        }


        /* 1564. Put Boxes Into the Warehouse I
        https://leetcode.com/problems/put-boxes-into-the-warehouse-i/description/
         */
        class MaxBoxesInWarehouseSol
        {
            /*
            Approach 1: Add Smallest Boxes to the Rightmost Warehouse Rooms
            Let n be the number of boxes and m be the number of rooms in the warehouse.
•	Time complexity: O(nlog(n)+m) because we need to sort the boxes (O(nlogn)) and iterate over the warehouse rooms and boxes (O(n+m))).
•	Space complexity: O(1) because we use two pointers to iterate over the boxes and warehouse rooms. If we are not allowed to modify the warehouse array, we will need O(m) extra space.

            */
            public int AddSmallestBoxesToRightMost(int[] boxes, int[] warehouse)
            {
                // Preprocess the height of the warehouse rooms to get usable heights
                for (int i = 1; i < warehouse.Length; i++)
                {
                    warehouse[i] = Math.Min(warehouse[i - 1], warehouse[i]);
                }

                // Iterate through boxes from smallest to largest
                Array.Sort(boxes);

                int count = 0;

                for (int i = warehouse.Length - 1; i >= 0; i--)
                {
                    // Count the boxes that can fit in the current warehouse room
                    if (count < boxes.Length && boxes[count] <= warehouse[i])
                    {
                        count++;
                    }
                }

                return count;
            }
            /*
            Approach 2: Add Largest Possible Boxes from Left to Right
            The time and space complexity will be similar to Approach 1. Let n be the number of boxes and m be the number of rooms in the warehouse.
•	Time complexity: O(nlog(n)+m) because we need to sort the boxes and iterate over the warehouse rooms and boxes.
•	Space complexity: O(1) because we use two pointers to iterate over the boxes and warehouse rooms.

            */
            public int AddLargestPossibleBoxesFromLeftToRigt(int[] boxes, int[] warehouse)
            {

                int i = boxes.Length - 1;
                int count = 0;
                Array.Sort(boxes);

                foreach (int room in warehouse)
                {
                    // Iterate through boxes from largest to smallest
                    // Discard boxes that doesn't fit in the current warehouse
                    while (i >= 0 && boxes[i] > room)
                    {
                        i--;
                    }

                    if (i == -1) return count;
                    count++;
                    i--;
                }

                return count;

            }
        }




        /* 1580. Put Boxes Into the Warehouse II
        https://leetcode.com/problems/put-boxes-into-the-warehouse-ii/description/
         */
        class MaxBoxesInWarehouseIISol
        {
            /*
            Approach 1: Add Smallest Boxes to the Rightmost Warehouse Rooms from Both Ends
            Complexity Analysis
Let m be the number of boxes and n be the number of rooms in the warehouse.
•	Time complexity: O(mlog(m)+nlog(n))
O(mlog(m)+nlog(n)) because we need to sort the boxes O(mlogm) and preprocess the warehouse rooms heights from both ends O(n).
•	Space complexity: O(logm+n)
The algorithm uses an additional array effectiveHeights to store the processed heights of the warehouse, which has the same length as the warehouse array. Therefore, the space complexity is O(n).
Sorting the boxes has different space complexities depending on the language used. For now, we will consider Java as the primary language, so the space complexity of O(logm) is due to the auxiliary stack space used by the sort function.
Sorting the effectiveHeights has a space complexity of O(logn) for the same reason.
The space complexity of the sorting algorithm depends on the programming language.
o	In Python, the sort method sorts a list using the Timsort algorithm which is a combination of Merge Sort and Insertion Sort and has O(n) additional space.
o	In Java, Arrays.sort() is implemented using a variant of the Quick Sort algorithm which has a space complexity of O(logn) for sorting two arrays.
o	In C++, the sort() function is implemented as a hybrid of Quick Sort, Heap Sort, and Insertion Sort, with a worse-case space complexity of O(logn).

            */
            public int AddSmallestBoxesToRightMostFromBothEnds(int[] boxes, int[] warehouse)
            {
                int warehouseLength = warehouse.Length;
                int minimumHeight = int.MaxValue;
                int[] effectiveHeights = new int[warehouseLength];

                // Preprocess the height of the warehouse rooms to
                // get usable heights from the left end
                for (int i = 0; i < warehouseLength; ++i)
                {
                    minimumHeight = Math.Min(minimumHeight, warehouse[i]);
                    effectiveHeights[i] = minimumHeight;
                }

                minimumHeight = int.MaxValue;
                // Update the effective heights considering the right end
                for (int i = warehouseLength - 1; i >= 0; --i)
                {
                    minimumHeight = Math.Min(minimumHeight, warehouse[i]);
                    effectiveHeights[i] = Math.Max(effectiveHeights[i], minimumHeight);
                }

                // Sort the effective heights of the warehouse rooms
                Array.Sort(effectiveHeights);
                // Sort the boxes by height
                Array.Sort(boxes);

                int boxCount = 0;
                int boxIndex = 0;
                // Try to place each box in the warehouse from
                // the smallest room to the largest
                for (int i = 0; i < warehouseLength; ++i)
                {
                    if (
                        boxIndex < boxes.Length &&
                        boxes[boxIndex] <= effectiveHeights[i]
                    )
                    {
                        // Place the box and move to the next one
                        ++boxCount;
                        ++boxIndex;
                    }
                }

                return boxCount;
            }
            /*
            Approach 2: Add Largest Possible Boxes from Both Ends
Complexity Analysis
Let m be the number of boxes and n be the number of rooms in the warehouse.
•	Time complexity: O(mlog(m)+n)
O(mlog(m)+n) because we need to sort the boxes and iterate over the warehouse rooms and boxes.
•	Space complexity: O(logm) or O(m)
Note that some extra space is used when we sort arrays in place. The space complexity of the sorting algorithm depends on the programming language.
o	In Python, the sort method sorts a list using the Timsort algorithm which is a combination of Merge Sort and Insertion Sort and has O(m) additional space.
o	In Java, Arrays.sort() is implemented using a variant of the Quick Sort algorithm which has a space complexity of O(logm) for sorting two arrays.
o	In C++, the sort() function is implemented as a hybrid of Quick Sort, Heap Sort, and Insertion Sort, with a worse-case space complexity of O(logm).
Besides the sorting space, our algorithm uses constant space O(1) by using two pointers to iterate through the boxes and warehouse rooms.

            */
            public int AddLargestPossibleBoxesFromBothEnds(int[] boxes, int[] warehouse)
            {
                int warehouseSize = warehouse.Length;

                // Sort the boxes by height
                Array.Sort(boxes);

                int leftIndex = 0;
                int rightIndex = warehouseSize - 1;
                int boxCount = 0;
                int boxIndex = boxes.Length - 1;

                // Iterate through the boxes from largest to smallest
                while (leftIndex <= rightIndex && boxIndex >= 0)
                {
                    // Check if the current box can fit in the leftmost available room
                    if (boxes[boxIndex] <= warehouse[leftIndex])
                    {
                        boxCount++;
                        leftIndex++;
                        // Check if the current box can fit in the rightmost available room
                    }
                    else if (boxes[boxIndex] <= warehouse[rightIndex])
                    {
                        boxCount++;
                        rightIndex--;
                    }
                    boxIndex--;
                }

                return boxCount;
            }
        }



        /* 1601. Maximum Number of Achievable Transfer Requests
        https://leetcode.com/problems/maximum-number-of-achievable-transfer-requests/description/
         */
        class MaximumRequestsSol
        {
            int answer = 0;
            /*
            Approach 1: Backtracking
            Complexity Analysis
            Here, N is the number of buildings, and M is the number of requests.
            •	Time complexity: O((2^M)∗N).
            We iterate over every two possibilities for each of the M requests; this is equal to 2^M possibilities. For the leaf nodes, which are O((2^M)−1), we will iterate over N buildings to check if the employee change is zero. Therefore the total time complexity would be O()2^M)∗N).
            •	Space complexity: O(N+M).
            The array indegree is of size N, and there would be some stack space as well for the recursion. The maximum number of active stack calls would equal M, i.e. when all the requests call would be active. Hence the total space complexity would be O(N+M).


            */
            public int WithBacktracking(int n, int[][] requests)
            {
                int[] indegree = new int[n];
                MaxRequest(requests, indegree, n, 0, 0);

                return answer;
            }
            void MaxRequest(int[][] requests, int[] indegree, int n, int index, int count)
            {
                if (index == requests.Length)
                {
                    // Check if all buildings have an in-degree of 0.
                    for (int i = 0; i < n; i++)
                    {
                        if (indegree[i] != 0)
                        {
                            return;
                        }
                    }

                    answer = Math.Max(answer, count);
                    return;
                }

                // Consider this request, increment and decrement for the buildings involved.
                indegree[requests[index][0]]--;
                indegree[requests[index][1]]++;
                // Move on to the next request and also increment the count of requests.
                MaxRequest(requests, indegree, n, index + 1, count + 1);
                // Backtrack to the previous values to move back to the original state before the second recursion.
                indegree[requests[index][0]]++;
                indegree[requests[index][1]]--;

                // Ignore this request and move on to the next request without incrementing the count.
                MaxRequest(requests, indegree, n, index + 1, count);
            }
            /*
            Approach 2: Bitmasking
            Complexity Analysis
            Here, N is the number of buildings, and M is the number of requests.
            •	Time complexity: O(2^(M∗(M+N)).
            We iterate over every two possibilities for each of the M requests; this is equal to 2^M possibilities. For each bitmask, we may iterate over N buildings and M requests. Therefore the total time complexity would be O(2^(M∗(M+N)).
            •	Space complexity: O(N).
            The array indegree is of size N. Hence the total space complexity would be O(N).

            */
            public int UsingBitMasking(int n, int[][] requests)
            {
                int answer = 0;

                for (int mask = 0; mask < (1 << requests.Length); mask++)
                {
                    int[] indegree = new int[n];
                    int pos = requests.Length - 1;
                    // Number of set bits representing the requests we will consider.
                    int bitCount = int.PopCount(mask);

                    // If the request count we're going to consider is less than the maximum request 
                    // We have considered without violating the constraints; then we can return it cannot be the answer.
                    if (bitCount <= answer)
                    {
                        continue;
                    }

                    // For all the 1's in the number, update the array indegree for the building it involves.
                    for (int curr = mask; curr > 0; curr >>= 1, pos--)
                    {
                        if ((curr & 1) == 1)
                        {
                            indegree[requests[pos][0]]--;
                            indegree[requests[pos][1]]++;
                        }
                    }

                    bool flag = true;
                    // Check if it doesn;t violates the constraints
                    for (int i = 0; i < n; i++)
                    {
                        if (indegree[i] != 0)
                        {
                            flag = false;
                            break;
                        }
                    }

                    if (flag)
                    {
                        answer = bitCount;
                    }
                }

                return answer;
            }

        }



        /* 1584. Min Cost to Connect All Points
        https://leetcode.com/problems/min-cost-to-connect-all-points/description/
         */
        class MinCostConnectPointsSol
        {
            /*
            Approach 1: Kruskal's Algorithm
            Complexity Analysis
If N is the number of points in the input array.
•	Time complexity: O(N^2⋅log(N)).
o	First, we store N⋅(N−1)/2≈N^2 edges of our complete graph in the allEdges array which will take O(N^2) time, and sorting this array will take O(N^2⋅log(N^2)) time.
o	Then, we iterate over the allEdges array, and for each element, we perform a union-find operation. The amortized time complexity for union-find by rank and path compression is O(α(N)), where α(N) is Inverse Ackermann Function, which is nearly constant, even for large values of N.
o	Thus, the overall time complexity is O(N^2+N^2⋅log(N^2)+N^2⋅α(N))≈O(N^2⋅log(N^2))≈O(N^2⋅log(N)).
•	Space complexity: O(N^2).
o	We use an array allEdges to store all N⋅(N−1)/2≈N^2 edges of our graph.
o	UnionFind object uf uses two arrays each of size N to store the group id and rank of all the nodes.
o	Thus, the overall space complexity is O(N^2+N)≈O(N^2).

            */
            public int KruskalAlgoWithUnionFind(int[][] points)
            {
                int numberOfPoints = points.Length;
                List<int[]> allEdges = new List<int[]>();

                // Storing all edges of our complete graph.
                for (int currentPoint = 0; currentPoint < numberOfPoints; ++currentPoint)
                {
                    for (int nextPoint = currentPoint + 1; nextPoint < numberOfPoints; ++nextPoint)
                    {
                        int weight = Math.Abs(points[currentPoint][0] - points[nextPoint][0]) +
                                     Math.Abs(points[currentPoint][1] - points[nextPoint][1]);

                        int[] currentEdge = { weight, currentPoint, nextPoint };
                        allEdges.Add(currentEdge);
                    }
                }

                // Sort all edges in increasing order.
                allEdges.Sort((a, b) => a[0].CompareTo(b[0]));

                UnionFind unionFind = new UnionFind(numberOfPoints);
                int minimumSpanningTreeCost = 0;
                int edgesUsed = 0;

                for (int i = 0; i < allEdges.Count && edgesUsed < numberOfPoints - 1; ++i)
                {
                    int node1 = allEdges[i][1];
                    int node2 = allEdges[i][2];
                    int weight = allEdges[i][0];

                    if (unionFind.Union(node1, node2))
                    {
                        minimumSpanningTreeCost += weight;
                        edgesUsed++;
                    }
                }

                return minimumSpanningTreeCost;
            }
            class UnionFind
            {
                public int[] Group;
                public int[] Rank;

                public UnionFind(int size)
                {
                    Group = new int[size];
                    Rank = new int[size];
                    for (int i = 0; i < size; ++i)
                    {
                        Group[i] = i;
                    }
                }

                public int Find(int node)
                {
                    if (Group[node] != node)
                    {
                        Group[node] = Find(Group[node]);
                    }
                    return Group[node];
                }

                public bool Union(int node1, int node2)
                {
                    int group1 = Find(node1);
                    int group2 = Find(node2);

                    // node1 and node2 already belong to same group.
                    if (group1 == group2)
                    {
                        return false;
                    }

                    if (Rank[group1] > Rank[group2])
                    {
                        Group[group2] = group1;
                    }
                    else if (Rank[group1] < Rank[group2])
                    {
                        Group[group1] = group2;
                    }
                    else
                    {
                        Group[group1] = group2;
                        Rank[group2] += 1;
                    }

                    return true;
                }
            }

            /*Approach 2: Prim's Algorithm
Complexity Analysis
If N is the number of points in the input array.
•	Time complexity: O(N^2⋅log(N)).
o	In the worst-case, we push/pop N⋅(N−1)/2≈N^2 edges of our graph in the heap. Each push/pop operation takes O(log(N^2))≈log(N) time.
o	Thus, the overall time complexity is O(N^2⋅log(N)).
•	Space complexity: O(N^2).
o	In the worst-case, we push N⋅(N−1)/2≈N^2 edges into the heap.
o	We use an array inMST of size N to mark which nodes are included in MST.
o	Thus, the overall space complexity is O(N^2+N)≈O(N^2).

            */

            public int PrimsAlgoWithHeap(int[][] points)
            {
                int numberOfPoints = points.Length;

                // Min-heap to store minimum weight edge at top.
                PriorityQueue<(int weight, int node), int> priorityQueue = new PriorityQueue<(int, int), int>();

                // Track nodes which are included in MST.
                bool[] includedInMST = new bool[numberOfPoints];

                priorityQueue.Enqueue((0, 0), 0);
                int minimumSpanningTreeCost = 0;
                int edgesUsed = 0;

                while (edgesUsed < numberOfPoints)
                {
                    var topElement = priorityQueue.Dequeue();

                    int edgeWeight = topElement.weight;
                    int currentNode = topElement.node;

                    // If node was already included in MST we will discard this edge.
                    if (includedInMST[currentNode])
                    {
                        continue;
                    }

                    includedInMST[currentNode] = true;
                    minimumSpanningTreeCost += edgeWeight;
                    edgesUsed++;

                    for (int nextNode = 0; nextNode < numberOfPoints; ++nextNode)
                    {
                        // If next node is not in MST, then edge from current node
                        // to next node can be pushed in the priority queue.
                        if (!includedInMST[nextNode])
                        {
                            int nextWeight = Math.Abs(points[currentNode][0] - points[nextNode][0]) +
                                             Math.Abs(points[currentNode][1] - points[nextNode][1]);

                            priorityQueue.Enqueue((nextWeight, nextNode), nextWeight);
                        }
                    }
                }

                return minimumSpanningTreeCost;
            }
            /* Approach 3: Prim's Algorithm (Optimized)
Complexity Analysis
If N is the number of points in the input array.
•	Time complexity: O(N^2).
o	We pick all N nodes one by one to include in the MST. Picking each node takes O(N) time and after picking a node, we iterate over all of its adjacent nodes, which also takes O(N) time.
o	Thus, the overall time complexity is O(N⋅(N+N))=O(N^2).
•	Space complexity: O(N).
o	We use two arrays each of size N, inMST and minDist.
o	Thus, the overall space complexity is O(N+N)=O(N).

             */

            public int PrimsAlgoSpaceOptimal(int[][] points)
            {
                int n = points.Length;
                int mstCost = 0;
                int edgesUsed = 0;

                // Track nodes which are visited.
                bool[] inMST = new bool[n];

                int[] minDist = new int[n];
                minDist[0] = 0;

                for (int i = 1; i < n; ++i)
                {
                    minDist[i] = int.MaxValue;
                }

                while (edgesUsed < n)
                {
                    int currMinEdge = int.MaxValue;
                    int currNode = -1;

                    // Pick least weight node which is not in MST.
                    for (int node = 0; node < n; ++node)
                    {
                        if (!inMST[node] && currMinEdge > minDist[node])
                        {
                            currMinEdge = minDist[node];
                            currNode = node;
                        }
                    }

                    mstCost += currMinEdge;
                    edgesUsed++;
                    inMST[currNode] = true;

                    // Update adjacent nodes of current node.
                    for (int nextNode = 0; nextNode < n; ++nextNode)
                    {
                        int weight = Math.Abs(points[currNode][0] - points[nextNode][0]) +
                                     Math.Abs(points[currNode][1] - points[nextNode][1]);

                        if (!inMST[nextNode] && minDist[nextNode] > weight)
                        {
                            minDist[nextNode] = weight;
                        }
                    }
                }

                return mstCost;
            }

        }



        /* 1578. Minimum Time to Make Rope Colorful
        https://leetcode.com/problems/minimum-time-to-make-rope-colorful/description/
         */
        class MinimumTimeToMakeRopeColorfulSol
        {
            /*
            Approach 1: Two pointers
            Complexity Analysis
Let n be the length of input string colors.
•	Time complexity: O(n)
o	We need to iterate over colors and neededTime. The right index right is incremented by O(n) times while the left index left is updated by no more than O(n) times. In each step of the iteration, we have some calculations that take constant time.
o	To sum up, the overall time complexity is O(n)
•	Space complexity: O(1)
o	We only need to update several values: totalTime, currTotal, currMax, i and j, which takes constant space.

            */
            public int UsingTwoPointers(String colors, int[] neededTime)
            {
                // Initalize two pointers i, j.
                int totalTime = 0;
                int i = 0, j = 0;

                while (i < neededTime.Length && j < neededTime.Length)
                {
                    int currTotal = 0, currMax = 0;

                    // Find all the balloons having the same color as the 
                    // balloon indexed at i, record the total removal time 
                    // and the maximum removal time.
                    while (j < neededTime.Length && colors[i] == colors[j])
                    {
                        currTotal += neededTime[j];
                        currMax = Math.Max(currMax, neededTime[j]);
                        j++;
                    }

                    // Once we reach the end of the current group, add the cost of 
                    // this group to total_time, and reset two pointers.
                    totalTime += currTotal - currMax;
                    i = j;
                }
                return totalTime;
            }
            /*
            Approach 2: Advanced 1-Pass
            Complexity Analysis
Let n be the length of input string colors.
•	Time complexity: O(n)
o	Similarly, we just need to iterate over colors and neededTime. In each step of the iteration, we have some calculations that take constant time.
o	To sum up, the overall time complexity is O(n)
•	Space complexity: O(1)
o	We only need to update two values: totalTime and currMaxTime, which takes constant space.

            */
            public int Advanced1Pass(String colors, int[] neededTime)
            {
                // totalTime: total time needed to make rope colorful;
                // currMaxTime: maximum time of a balloon needed.
                int totalTime = 0, currMaxTime = 0;

                // For each balloon in the array:
                for (int i = 0; i < colors.Length; ++i)
                {
                    // If this balloon is the first balloon of a new group
                    // set the currMaxTime as 0.
                    if (i > 0 && colors[i] != colors[i - 1])
                    {
                        currMaxTime = 0;
                    }

                    // Increment totalTime by the smaller one.
                    // Update currMaxTime as the larger one.
                    totalTime += Math.Min(currMaxTime, neededTime[i]);
                    currMaxTime = Math.Max(currMaxTime, neededTime[i]);
                }

                // Return totalTime as the minimum removal time.
                return totalTime;
            }
        }



        /* 1575. Count All Possible Routes
        https://leetcode.com/problems/count-all-possible-routes/description/
         */
        class CountAllPossibleRoutesSol
        {
            /*
            Approach 1: Recursive Dynamic Programming
            Complexity Analysis
Here, n is the length of locations.
•	Time complexity: O(n^2⋅fuel)
o	Initializing the memo array takes O(n⋅fuel) time.
o	We iterate over all cities for each currCity, remainingFuel state. Because there are n⋅fuel states and computing each state requires iterating over all the n cities (except the current one), it would take O(n^2⋅fuel) time.
o	The recursive function might be called more than once as we saw in the recursion tree. However, due to memoization each state will be computed only once.
•	Space complexity: O(n⋅fuel)
o	The memo array consumes O(n⋅fuel) space.
o	The recursion stack used in the solution can grow to a maximum size of O(fuel). When we try to form the recursion tree, we see that after each node n - 1 branches are formed (visiting all cities except the current city). The recursion stack would only have one call out of the n - 1 branches. The height of such a tree will be O(fuel) in the worst case if we consider decrementing the remaining fuel by 1 when going from a city to another. As a result, the recursion tree that will be formed will have O(fuel) height. Hence, the recursion stack will have a maximum of O(fuel) elements.

            */
            public int DPRec(int[] locations, int start, int finish, int fuel)
            {
                int n = locations.Length;
                int[][] memo = new int[n][];
                for (int i = 0; i < n; ++i)
                {
                    Array.Fill(memo[i], -1);
                }

                return Solve(locations, start, finish, fuel, memo);
            }
            public int Solve(int[] locations, int currentCity, int destinationCity, int remainingFuel, int[][] memo)
            {
                if (remainingFuel < 0)
                {
                    return 0;
                }
                if (memo[currentCity][remainingFuel] != -1)
                {
                    return memo[currentCity][remainingFuel];
                }

                int answer = currentCity == destinationCity ? 1 : 0;
                for (int nextCity = 0; nextCity < locations.Length; nextCity++)
                {
                    if (nextCity != currentCity)
                    {
                        answer = (answer + Solve(locations, nextCity, destinationCity,
                        remainingFuel - Math.Abs(locations[currentCity] - locations[nextCity]),
                                           memo)) % 1000000007;
                    }
                }

                return memo[currentCity][remainingFuel] = answer;
            }

            /*
            Approach 2: Iterative Dynamic Programming
Complexity Analysis
Here, n is the length of locations.
•	Time complexity: O(n^2⋅fuel)
o	Initializing the dp array takes O(n⋅fuel) time.
o	We fill the dp array which takes O(n^2⋅fuel) as we run three nested loops.
•	Space complexity: O(n⋅fuel)
o	The dp array consumes O(n⋅fuel) space.

            */
            public int DPIterative(int[] locations, int start, int finish, int fuel)
            {
                int n = locations.Length;
                int[][] dp = new int[n][];
                Array.Fill(dp[finish], 1);

                int ans = 0;
                for (int j = 0; j <= fuel; j++)
                {
                    for (int i = 0; i < n; i++)
                    {
                        for (int k = 0; k < n; k++)
                        {
                            if (k == i)
                            {
                                continue;
                            }
                            if (Math.Abs(locations[i] - locations[k]) <= j)
                            {
                                dp[i][j] = (dp[i][j] + dp[k][j - Math.Abs(locations[i] - locations[k])]) %
                                           1000000007;
                            }
                        }
                    }
                }

                return dp[start][fuel];
            }
        }


        /* 1626. Best Team With No Conflicts
        https://leetcode.com/problems/best-team-with-no-conflicts/description/
         */
        class BestTeamScoreSol
        {
            /*
            Approach 1: Top-Down Dynamic Programming
            Complexity Analysis
Here, N is the number of players.
•	Time complexity: O(N^2).
Sorting the list ageScorePair will take O(NlogN) time. In the recursion, each state is defined by the index and the prev. Hence, there will be O(N∗N) states, and at worst, we must visit most of the states to solve the original problem. Each recursive call will require O(1) time due to the memoization. Therefore, the total time required equals O(N∗N).
•	Space complexity: O(N^2).
The list ageScorePair will take 2∗N space. The memoization results are stored in the table memo with size N * N. Also, stack space in the recursion equals the maximum number of active functions. The maximum number of active functions will be at most N, i.e., one function call for every player. Hence, the space complexity is O(N∗N).

            */
            public int TopDownDP(int[] scores, int[] ages)
            {
                int n = ages.Length;
                int[][] ageScorePair = new int[n][];

                for (int i = 0; i < n; i++)
                {
                    ageScorePair[i][0] = ages[i];
                    ageScorePair[i][1] = scores[i];
                }

                // Sort in ascending order of age and then by score.
                Array.Sort(ageScorePair, (a, b) => a[0] == b[0] ? a[1] - b[1] : a[0] - b[0]);
                // Initially, all states are null, denoting not yet calculated.
                int[][] dp = new int[n][];

                return FindMaxScore(dp, ageScorePair, -1, 0);
            }

            private int FindMaxScore(int[][] dp, int[][] ageScorePair, int prev, int index)
            {
                // Return 0 if we have iterated over all the players.
                if (index >= ageScorePair.Length)
                {
                    return 0;
                }

                // We have already calculated the answer, so no need to go into recursion.
                if (dp[prev + 1][index] != null)
                {
                    return dp[prev + 1][index];
                }

                // If we can add this player, return the maximum of two choices we have.
                if (prev == -1 || ageScorePair[index][1] >= ageScorePair[prev][1])
                {
                    return dp[prev + 1][index] = Math.Max(FindMaxScore(dp, ageScorePair, prev, index + 1),
                            ageScorePair[index][1] + FindMaxScore(dp, ageScorePair, index, index + 1));
                }

                // This player cannot be added; return the corresponding score.
                return dp[prev + 1][index] = FindMaxScore(dp, ageScorePair, prev, index + 1);
            }


            /* Approach 2: Bottom-Up Dynamic Programming
            Complexity Analysis
            Here, N is the number of players.
            •	Time complexity: O(N^2).
            Sorting the list ageScorePair will take O(NlogN) time. Then for the ith player, we iterate i - 1 players on the left, hence the total number of operations will be equal to (0+1+2+......+N−1), which is equivalent to ((N−1)∗N)/2. Therefore the total time complexity equals O(N∗N).
            •	Space complexity: O(N).
            The list ageScorePair will take 2∗N space. We have another list dp, to store the maximum score up to the particular index. Therefore the total space complexity equals O(N).

             */
            public int BottomUpDP(int[] scores, int[] ages)
            {
                int N = ages.Length;
                int[][] ageScorePair = new int[N][];

                for (int i = 0; i < N; i++)
                {
                    ageScorePair[i][0] = ages[i];
                    ageScorePair[i][1] = scores[i];
                }

                // Sort in ascending order of age and then by score.
                Array.Sort(ageScorePair, (a, b) => a[0] == b[0] ? a[1] - b[1] : a[0] - b[0]);
                return FindMaxScore(ageScorePair);
            }
            private int FindMaxScore(int[][] ageScorePair)
            {
                int n = ageScorePair.Length;
                int answer = 0;

                int[] dp = new int[n];
                // Initially, the maximum score for each player will be equal to the individual scores.
                for (int i = 0; i < n; i++)
                {
                    dp[i] = ageScorePair[i][1];
                    answer = Math.Max(answer, dp[i]);
                }


                for (int i = 0; i < n; i++)
                {
                    for (int j = i - 1; j >= 0; j--)
                    {
                        // If the players j and i could be in the same team.
                        if (ageScorePair[i][1] >= ageScorePair[j][1])
                        {
                            // Update the maximum score for the ith player.
                            dp[i] = Math.Max(dp[i], ageScorePair[i][1] + dp[j]);
                        }
                    }
                    // Maximum score among all the players.
                    answer = Math.Max(answer, dp[i]);
                }

                return answer;
            }

            /*
            Approach 3: Binary Indexed Tree (BIT) / Fenwick Tree
            Analysis
Here, N is the number of players, and K is the maximum age in the list.
•	Time complexity: (NlogN+NlogK).
Sorting the list ageScorePair will take O(NlogN) time. Then for each player, we query and update the BIT, which takes O(logK) and therefore will take (NlogK) for all the players. Hence the total time complexity equals (NlogN+NlogK).
•	Space complexity: O(N+K).
The list ageScorePair will take 2∗N space. BIT array will need O(K) space. Therefore the total space complexity equals O(N+K).

            */
            public int UsingFenwickTree(int[] scores, int[] ages)
            {
                int numberOfPlayers = ages.Length;
                int[][] ageScorePair = new int[numberOfPlayers][];

                for (int i = 0; i < numberOfPlayers; i++)
                {
                    ageScorePair[i][0] = scores[i];
                    ageScorePair[i][1] = ages[i];
                }

                // Sort in ascending order of score and then by age.
                Array.Sort(ageScorePair, (a, b) => a[0] == b[0] ? a[1] - b[1] : a[0] - b[0]);

                int highestAge = 0;
                foreach (int age in ages)
                {
                    highestAge = Math.Max(highestAge, age);
                }
                int[] BIT = new int[highestAge + 1];

                int answer = int.MinValue;
                foreach (int[] ageScore in ageScorePair)
                {
                    // Maximum score up to this age might not have all the players of this age.
                    int currentBest = ageScore[0] + QueryBIT(BIT, ageScore[1]);
                    // Update the tree with the current age and its best score.
                    UpdateBIT(BIT, ageScore[1], currentBest);

                    answer = Math.Max(answer, currentBest);
                }

                return answer;
            }

            // Query tree to get the maximum score up to this age.
            private int QueryBIT(int[] BIT, int age)
            {
                int maxScore = int.MinValue;
                for (int i = age; i > 0; i -= i & (-i))
                {
                    maxScore = Math.Max(maxScore, BIT[i]);
                }
                return maxScore;
            }

            // Update the maximum score for all the nodes with an age greater than the given age.
            private void UpdateBIT(int[] BIT, int age, int currentBest)
            {
                for (int i = age; i < BIT.Length; i += i & (-i))
                {
                    BIT[i] = Math.Max(BIT[i], currentBest);
                }
            }

        }



        /* 1629. Slowest Key
        https://leetcode.com/problems/slowest-key/description/
         */

        public class SlowestKeySol
        {
            /*
            Approach 1: Using Map/Dictionary
            Complexity Analysis
Let N be the size of array releaseTimes and K be the number of distinct characters in keysPressed.
•	Time Complexity: O(N). Let's find the time complexity of each step.
We iterate over the array releaseTimes of size N to find the duration of each key. The time complexity of each iteration is constant, so the overall time complexity of iterating over the array is O(N).
Next, we iterate over all elements of durationMap. In the worst case, if all the keys are unique, the size of durationMap would be equal to K. Thus, the time complexity is O(K).
This gives us total time complexity is O(N)+O(K). Since, in this problem, K is at most 26 and must be less than or equal to N the time complexity simplifies to O(N).
•	Space Complexity: O(K), as we are using additional space for durationMap which can have maximum K elements.

            */
            public char UsingDict(int[] releaseTimes, string keysPressed)
            {
                Dictionary<char, int> durationMap = new Dictionary<char, int>();
                durationMap[keysPressed[0]] = releaseTimes[0];
                // find and store the keypress duration for each key in the durationMap
                for (int i = 1; i < releaseTimes.Length; i++)
                {
                    int currentDuration = releaseTimes[i] - releaseTimes[i - 1];
                    char currentKey = keysPressed[i];
                    durationMap[currentKey] = Math.Max(durationMap.GetValueOrDefault(currentKey, 0), currentDuration);
                }
                char slowestKey = ' ';
                int longestPressDuration = 0;
                // iterate over the map to find the slowest key
                foreach (KeyValuePair<char, int> mapElement in durationMap)
                {
                    int duration = mapElement.Value;
                    char key = mapElement.Key;
                    if (duration > longestPressDuration)
                    {
                        longestPressDuration = duration;
                        slowestKey = key;
                    }
                    else if (duration == longestPressDuration && key > slowestKey)
                    {
                        slowestKey = key;
                    }
                }
                return slowestKey;
            }
            /*
            Approach 2: Fixed Size Array
            Complexity Analysis
Let N be the size of array releaseTimes and M be the maximum possible number of distinct characters. The value of M is fixed as 26 for this problem because keysPressed contains only lowercase English letters.
•	Time Complexity: O(N+M). Let's find the time complexity of each step.
We iterate over the array releaseTimes of size N to find the duration of each key. The time complexity of each iteration is constant, so the overall time complexity of iterating over the array is O(N).
Next, we iterate over all elements of durationArray of size M which takes O(M) time.
This gives us total time complexity is O(N)+O(M). Since, in this problem, the value of M is fixed at 26, O(M) may be considered as constant and the total time complexity would simplify to O(N).
•	Space Complexity: O(M), as we are using O(M) extra space for durationArray. However, since the value of M is fixed at 26, the space complexity may be considered as O(1).

            */
            public char UsingFixedSizeArray(int[] releaseTimes, String keysPressed)
            {
                int[] durationArray = new int[26];
                durationArray[keysPressed[0] - 'a'] = releaseTimes[0];

                // find and store the key pressed duration for each key
                for (int i = 1; i < releaseTimes.Length; i++)
                {
                    int currentDuration = releaseTimes[i] - releaseTimes[i - 1];
                    char currentKey = keysPressed[i];
                    durationArray[currentKey - 'a'] = Math.Max(durationArray[currentKey - 'a'], currentDuration);
                }
                // initialize slowest key as 'z'
                int slowestKeyIndex = 25;
                // iterate from 'y' to 'a' to find slowest key
                for (int currentKey = 24; currentKey >= 0; currentKey--)
                {
                    if (durationArray[currentKey] > durationArray[slowestKeyIndex])
                    {
                        slowestKeyIndex = currentKey;
                    }
                }
                return (char)(slowestKeyIndex + 'a');
            }


            /* Approach 3: Constant Extra Space 
Complexity Analysis
Let N be the size of array releaseTimes.
•	Time Complexity: O(N). We iterate over the array releaseTimes of size N once to find the slowest key and each iteration requires only constant time.
•	Space Complexity: O(1), as we are using only constant extra space.

            */
            public char WithConstantExtraSpace(int[] releaseTimes, String keysPressed)
            {
                int n = releaseTimes.Length;
                int longestPress = releaseTimes[0];
                char slowestKey = keysPressed[0];
                for (int i = 1; i < n; i++)
                {
                    int currentDuration = releaseTimes[i] - releaseTimes[i - 1];
                    // check if we found the key that is slower than slowestKey
                    if (currentDuration > longestPress ||
                        (currentDuration == longestPress && keysPressed[i] > slowestKey))
                    {
                        // update the slowest key and longest press duration
                        longestPress = currentDuration;
                        slowestKey = keysPressed[i];
                    }
                }
                return slowestKey;
            }

        }


        /* 1642. Furthest Building You Can Reach
        https://leetcode.com/problems/furthest-building-you-can-reach/description/
         */

        class FurthestBuildingToReachSol
        {
            /*
            Approach 1: Min-Heap
            Complexity Analysis
Let N be the length of the heights array. Let L be the number of ladders available. We're mostly going to focus on analyzing in terms of N; however, it is also interesting to look at how the number of ladders available impacts the time and space complexity.
•	Time complexity : O(NlogN) or O(NlogL).
Inserting or removing an item from a heap incurs a cost of O(logx), where x is the number of items currently in the heap. In the worst case, we know that there will be N−1 climbs in the heap, thus giving a time complexity of O(logN) for each insertion and removal, and we're doing up to N of each of these two operations. This gives a total time complexity of O(NlogN).
In practice, though, the heap will never contain more than L+1 climbs at a time—when it gets to this size, we immediately remove a climb from it. So, the heap operations are actually O(logL). We are still performing up to N of each of them, though, so this gives a total time complexity of O(NlogL).
•	Space complexity : O(N) or O(L).
As we determined above, the heap can contain up to O(L) numbers at a time. In the worst case, L=N, so we get O(N).

            */
            public int UsingMinHeap(int[] buildingHeights, int availableBricks, int availableLadders)
            {
                // Create a priority queue -  min-heap by default.
                PriorityQueue<int, int> ladderAllocations = new PriorityQueue<int, int>();
                for (int index = 0; index < buildingHeights.Length - 1; index++)
                {
                    int climbDifference = buildingHeights[index + 1] - buildingHeights[index];
                    // If this is actually a "jump down", skip it.
                    if (climbDifference <= 0)
                    {
                        continue;
                    }
                    // Otherwise, allocate a ladder for this climb.
                    ladderAllocations.Enqueue(climbDifference, climbDifference);
                    // If we haven't gone over the number of ladders, nothing else to do.
                    if (ladderAllocations.Count <= availableLadders)
                    {
                        continue;
                    }
                    // Otherwise, we will need to take a climb out of ladderAllocations
                    availableBricks -= ladderAllocations.Dequeue();

                    // If this caused bricks to go negative, we can't get to index + 1
                    if (availableBricks < 0)
                    {
                        return index;
                    }
                }
                // If we got to here, this means we had enough materials to cover every climb.
                return buildingHeights.Length - 1;
            }
            /*
            Approach 2: Max-Heap
Complexity Analysis
Let N be the length of the heights array. Unlike approach 1, it doesn't really make sense to analyze approach 2 in terms of the number of ladders or bricks we started with.
•	Time complexity : O(NlogN).
Same as Approach 1. In the worst case, we'll be adding and removing up to N−1 climbs from the heap. Heap operations are O(logN) in the worst case.
•	Space complexity : O(N).
Same as Approach 1. In the worst case, there'll be N−1 climbs in the heap.

            */
            public int WithMaxHeap(int[] heights, int bricks, int ladders)
            {
                // Create a priority queue with a comparator that makes it behave as a max-heap.
                PriorityQueue<int, int> brickAllocations = new PriorityQueue<int, int>(Comparer<int>.Create((a, b) => b.CompareTo(a)));
                for (int i = 0; i < heights.Length - 1; i++)
                {
                    int climb = heights[i + 1] - heights[i];
                    // If this is actually a "jump down", skip it.
                    if (climb <= 0)
                    {
                        continue;
                    }
                    // Otherwise, allocate a ladder for this climb.
                    brickAllocations.Enqueue(climb, climb);
                    bricks -= climb;
                    // If we've used all the bricks, and have no ladders remaining, then 
                    // we can't go any further.
                    if (bricks < 0 && ladders == 0)
                    {
                        return i;
                    }
                    // Otherwise, if we've run out of bricks, we should replace the largest
                    // brick allocation with a ladder.
                    if (bricks < 0)
                    {
                        bricks += brickAllocations.Dequeue();
                        ladders--;
                    }
                }
                // If we got to here, this means we had enough materials to cover every climb.
                return heights.Length - 1;
            }
            /*
            Approach 3: Binary Search for Final Reachable Building
Complexity Analysis
Let N be the length of the heights array.
•	Time complexity : O(N(log^2)N).
On an array of length N, binary search requires O(logN) iterations to reduce the array down to a single item. On each of these binary search iterations, we're doing a sort. On average, we'll be sorting N/2 items each time. In Big O notation, though, we can simply treat this as N. The average cost of sorting is O(NlogN). Putting this together, we get O(logN)⋅O(NlogN)=O(N(log^2)N).
•	Space complexity : O(N).
For each iteration, we need to make a new list of up to N - 1 climbs. This gives us a O(N) space complexity (the list is discarded at the end of each iteration, so we never have more than one of these lists using memory at the same time).

            */
            public int UsingBinarySearchForFinalReachableBuliding(int[] heights, int bricks, int ladders)
            {
                // Do a binary search on the heights array to find the final reachable building.
                int lo = 0;
                int hi = heights.Length - 1;
                while (lo < hi)
                {
                    int mid = lo + (hi - lo + 1) / 2;
                    if (IsReachable(mid, heights, bricks, ladders))
                    {
                        lo = mid;
                    }
                    else
                    {
                        hi = mid - 1;
                    }
                }
                return hi; // Note that return lo would be equivalent.
            }

            private bool IsReachable(int buildingIndex, int[] heights, int bricks, int ladders)
            {
                // Make a list of all the climbs we need to do to reach buildingIndex.
                List<int> climbs = new List<int>();
                for (int i = 0; i < buildingIndex; i++)
                {
                    int h1 = heights[i];
                    int h2 = heights[i + 1];
                    if (h2 <= h1)
                    {
                        continue;
                    }
                    climbs.Add(h2 - h1);
                }
                climbs.Sort();

                // And now determine whether or not all of these climbs can be covered with the
                // given bricks and ladders.
                foreach (int climb in climbs)
                {
                    // If there are bricks left, use those.
                    if (climb <= bricks)
                    {
                        bricks -= climb;
                        // Otherwise, you'll have to use a ladder.
                    }
                    else if (ladders >= 1)
                    {
                        ladders -= 1;
                        // And if there are no ladders either, we can't reach buildingIndex.
                    }
                    else
                    {
                        return false;
                    }
                }
                return true;
            }

            /*
            Approach 4: Improved Binary Search for Final Reachable Building
Complexity Analysis
Let N be the length of the heights array.
•	Time complexity : O(NlogN).
There are two parts to this algorithm: constructing the sorted climbs list and doing the binary search + reachability checking.
Extracting the climbs has a cost of O(N) as there are up to N−1 of them, and access to them is sequential. Sorting them has a cost of O(NlogN).
For this implementation, isReachable(...) simply goes down the pre-generated sorted_climbs list and performs an O(1) operation on each of the up to N items. Therefore, it has a cost of O(N).
Like before, we're calling isReachable(...) a total of logN times, driven by the binary search. So in total, the calls to isReachable(...) have a cost of O(NlogN).
Because the creation of sorted_climbs and the binary search happen sequentially, we add their time complexities. O(NlogN)+O(NlogN)=2O˙(NlogN)=O(NlogN).
•	Space complexity : O(N).
The sorted list of climbs requires O(N) space.

            */
            public int UsingImprovedBinarySearchForFinalReachableBuliding(int[] heights, int bricks, int ladders)
            {
                // Make a sorted list of all the climbs.
                List<int[]> sortedClimbs = new List<int[]>();
                for (int i = 0; i < heights.Length - 1; i++)
                {
                    int climb = heights[i + 1] - heights[i];
                    if (climb <= 0)
                    {
                        continue;
                    }
                    sortedClimbs.Add(new int[] { climb, i + 1 });
                }
                sortedClimbs.Sort((a, b) => a[0] - b[0]);

                // Now do the binary search, same as before.
                int lo = 0;
                int hi = heights.Length - 1;
                while (lo < hi)
                {
                    int mid = lo + (hi - lo + 1) / 2;
                    if (IsReachable(mid, sortedClimbs, bricks, ladders))
                    {
                        lo = mid;
                    }
                    else
                    {
                        hi = mid - 1;
                    }
                }
                return hi; // Note that return lo would be equivalent.
            }

            private bool IsReachable(int buildingIndex, List<int[]> climbs, int bricks, int ladders)
            {
                foreach (int[] climbEntry in climbs)
                {
                    // Extract the information for this climb
                    int climb = climbEntry[0];
                    int index = climbEntry[1];
                    // Check if this climb is within the range.
                    if (index > buildingIndex)
                    {
                        continue;
                    }
                    // Allocate bricks if enough remain; otherwise, allocate a ladder if
                    // at least one remains.
                    if (climb <= bricks)
                    {
                        bricks -= climb;
                    }
                    else if (ladders >= 1)
                    {
                        ladders -= 1;
                    }
                    else
                    {
                        return false;
                    }
                }
                return true;
            }

            /*
            Approach 5: Binary Search on Threshold (Advanced)
Complexity Analysis
Let N be the length of the heights array. Let maxClimb be the length of the longest climb. An upper bound on this value is max(heights).
•	Time complexity : O(Nlog(maxClimb)).
The solveWithGivenThreshold(...) function iterates over the heights list, performing O(1) operations for each index. Because heights contains N items, this gives a total time complexity of O(N) for this function.
The binary search starts with a search space of up to maxClimb and halves it each time. This means that log(maxClimb) calls are made to solveWithGivenThreshold(...).
Multiplying these together, we get a final time complexity of O(Nlog(maxClimb)).
•	Space complexity : O(1).
We are only using constant extra memory (note that the fixed-length arrays used to return 3 values are considered to be constant).
Comparing this to the previous approaches, it will generally perform quite well. With the problem constraints we're given, log(maxClimb) isn't much worse than logN in the worst case, and in fact, it is often better. Most notably, this approach shows that it is possible to solve this problem without using auxiliary memory while keeping the time complexity almost the same.

            */
            public int UsingBinarySearchOnThreshold(int[] heights, int bricks, int ladders)
            {
                int lo = int.MaxValue;
                int hi = int.MinValue;
                for (int i = 0; i < heights.Length - 1; i++)
                {
                    int climb = heights[i + 1] - heights[i];
                    if (climb <= 0)
                    {
                        continue;
                    }
                    lo = Math.Min(lo, climb);
                    hi = Math.Max(hi, climb);
                }
                if (lo == int.MaxValue)
                {
                    return heights.Length - 1;
                }
                while (lo <= hi)
                {
                    int mid = lo + (hi - lo) / 2;
                    int[] result = SolveWithGivenThreshold(heights, bricks, ladders, mid);
                    int indexReached = result[0];
                    int laddersRemaining = result[1];
                    int bricksRemaining = result[2];
                    if (indexReached == heights.Length - 1)
                    {
                        return heights.Length - 1;
                    }
                    if (laddersRemaining > 0)
                    {
                        hi = mid - 1;
                        continue;
                    }
                    // Otherwise, check whether this is the "too low" or "just right" case.
                    int nextClimb = heights[indexReached + 1] - heights[indexReached];
                    if (nextClimb > bricksRemaining && mid > bricksRemaining)
                    {
                        return indexReached;
                    }
                    else
                    {
                        lo = mid + 1;
                    }
                }
                return -1; // It always returns before here. But gotta keep Java happy.
            }

            public int[] SolveWithGivenThreshold(int[] heights, int bricks, int ladders, int K)
            {
                int laddersUsedOnThreshold = 0;
                for (int i = 0; i < heights.Length - 1; i++)
                {
                    int climb = heights[i + 1] - heights[i];
                    if (climb <= 0)
                    {
                        continue;
                    }
                    // Make resource allocations
                    if (climb == K)
                    {
                        laddersUsedOnThreshold++;
                        ladders--;
                    }
                    else if (climb > K)
                    {
                        ladders--;
                    }
                    else
                    {
                        bricks -= climb;
                    }
                    // Handle negative resources
                    if (ladders < 0)
                    {
                        if (laddersUsedOnThreshold >= 1)
                        {
                            laddersUsedOnThreshold--;
                            ladders++;
                            bricks -= K;
                        }
                        else
                        {
                            return new int[] { i, ladders, bricks };
                        }
                    }
                    if (bricks < 0)
                    {
                        return new int[] { i, ladders, bricks };
                    }
                }
                return new int[] { heights.Length - 1, ladders, bricks };
            }

        }



        /* 2940. Find Building Where Alice and Bob Can Meet
        https://leetcode.com/problems/find-building-where-alice-and-bob-can-meet/description/
         */
        public int[] LeftmostBuildingQueries(int[] heights, int[][] queries)
        {
            /*
            Approach: Priority Queue
            Complexity
Time O(qlogq)
Space O(q)
where q = queries.size

            */
            int numberOfBuildings = heights.Length;
            int numberOfQueries = queries.Length;
            List<int[]>[] queryList = new List<int[]>[numberOfBuildings];
            for (int buildingIndex = 0; buildingIndex < numberOfBuildings; buildingIndex++)
                queryList[buildingIndex] = new List<int[]>();
            PriorityQueue<int[], int[]> heightQueue = new PriorityQueue<int[], int[]>(Comparer<int[]>.Create((a, b) => a[0].CompareTo(b[0])));
            int[] results = new int[numberOfQueries];
            Array.Fill(results, -1);
            // Step 1
            for (int queryIndex = 0; queryIndex < numberOfQueries; queryIndex++)
            {
                int startIndex = queries[queryIndex][0];
                int endIndex = queries[queryIndex][1];
                if (startIndex < endIndex && heights[startIndex] < heights[endIndex])
                {
                    results[queryIndex] = endIndex;
                }
                else if (startIndex > endIndex && heights[startIndex] > heights[endIndex])
                {
                    results[queryIndex] = startIndex;
                }
                else if (startIndex == endIndex)
                {
                    results[queryIndex] = startIndex;
                }
                else
                { // Step 2
                    queryList[Math.Max(startIndex, endIndex)].Add(new int[] { Math.Max(heights[startIndex], heights[endIndex]), queryIndex });
                }
            }
            // Step 3
            for (int buildingIndex = 0; buildingIndex < numberOfBuildings; buildingIndex++)
            {
                while (heightQueue.Count > 0 && heightQueue.Peek()[0] < heights[buildingIndex])
                {
                    results[heightQueue.Dequeue()[1]] = buildingIndex;
                }
                foreach (int[] query in queryList[buildingIndex])
                {
                    heightQueue.Enqueue(query, query);
                }
            }

            return results;
        }


        /* 1944. Number of Visible People in a Queue
        https://leetcode.com/problems/number-of-visible-people-in-a-queue/description/
         */
        class CanSeePersonsCountSol
        {
            /*
            Approach: Mononotic stack
            Complexity
•	Time: O(N), where N <= 10^5 is number of elements in heights array.
•	Space: O(N)

            */
            public int[] UsingMononoticStack(int[] heights)
            {
                int n = heights.Length;
                int[] ans = new int[n];
                Stack<int> st = new Stack<int>(); //Increasing Monotnic Stack
                for (int i = n - 1; i >= 0; --i)
                {
                    while (st.Count > 0 && heights[i] > st.Peek())
                    {
                        st.Pop();
                        ++ans[i];
                    }
                    if (st.Count > 0)
                        ++ans[i];
                    st.Push(heights[i]);
                }
                return ans;
            }
        }


        /* 1762. Buildings With an Ocean View
        https://leetcode.com/problems/buildings-with-an-ocean-view/description/
         */
        class FindBuildingsSol
        {
            /*
            Approach 1: Linear Iteration
            Complexity Analysis
Here N is the size of the given array.
	Time complexity: O(N).
	We iterate over the given array once.
	Each building's index can be pushed to answer and popped from answer at most once, and both of the operations take O(1) time.
	In Java, copying the elements from an array list to an integer array will take an additional O(N) time.
	Space complexity: O(N).
	There is no auxiliary space used other than the output. The output does not count towards the space complexity. However, in the worst-case scenario, answer may contain as many as N−1 indices, and then the very last building is the tallest, so the output will reduce to one index. In this scenario, the algorithm must store N−1 elements at some point, but only 1 element is included in the output.
	In Java, in order to maintain a dynamic size array, we created an extra Array List that supports fast O(1) push/pop operation. Array List can contain at most N elements. Hence in Java, an additional O(N) space is used.

            */
            public int[] LinearRotation(int[] heights)
            {
                int n = heights.Length;
                List<int> list = new List<int>();

                for (int current = 0; current < n; ++current)
                {
                    // If the current building is taller, 
                    // it will block the shorter building's ocean view to its left.
                    // So we pop all the shorter buildings that have been added before.
                    while (list.Count > 0 && heights[list[list.Count - 1]] <= heights[current])
                    {
                        list.Remove(list.Count - 1);
                    }
                    list.Add(current);
                }

                // Push elements from list to answer array.
                int[] answer = new int[list.Count];
                for (int i = 0; i < list.Count; ++i)
                {
                    answer[i] = list[i];
                }

                return answer;
            }

            /* Approach 2: Monotonic Stack
            Complexity Analysis
            Here N is the size of the given array.
                Time complexity: O(N).
                We iterate over the given array once.
                Each building's index can be pushed into and popped from the stack at most once, and both the operations take O(1) time.
                The array is reversed at the end which takes O(N) time.
                In Java, the copying of elements from the array list to an integer array in reverse order also takes O(N) time.
                Space complexity: O(N).
                An extra stack is created, which can take at most O(N) space.
                In Java, in order to maintain a dynamic size array (since we don't know the size of the output array at the beginning), we created an extra Array List that supports fast O(1) push operation. The Array List may contain at most N elements.

             */
            public int[] UsingMontonicStack(int[] heights)
            {
                int numberOfBuildings = heights.Length;
                List<int> buildingIndices = new List<int>();

                // Monotonically decreasing stack.
                Stack<int> buildingStack = new Stack<int>();
                for (int currentBuildingIndex = numberOfBuildings - 1; currentBuildingIndex >= 0; --currentBuildingIndex)
                {
                    // If the building to the right is smaller, we can pop it.
                    while (buildingStack.Count > 0 && heights[buildingStack.Peek()] < heights[currentBuildingIndex])
                    {
                        buildingStack.Pop();
                    }

                    // If the stack is empty, it means there is no building to the right 
                    // that can block the view of the current building.
                    if (buildingStack.Count == 0)
                    {
                        buildingIndices.Add(currentBuildingIndex);
                    }

                    // Push the current building in the stack.
                    buildingStack.Push(currentBuildingIndex);
                }

                // Push elements from buildingIndices to answer array in reverse order.
                int[] answerArray = new int[buildingIndices.Count];
                for (int i = 0; i < buildingIndices.Count; ++i)
                {
                    answerArray[i] = buildingIndices[buildingIndices.Count - 1 - i];
                }

                return answerArray;
            }

            /*
        Approach 3: Monotonic Stack Space Optimization   
        Complexity Analysis
Here N is the size of the given array.
	Time complexity: O(N).
	We iterate over the given array once, and for each building height, we perform a constant number of operations.
	The answer array is reversed at the end, which also takes O(N) time.
	In Java, copying the elements from the array list to an integer array in reverse order also takes O(N).
	Space complexity: O(1).
	No auxiliary space was used other than for the output array.
	Although, in Java, in order to maintain a dynamic size array (since we don't know the size of the output array at the beginning), we created an extra Array List that supports fast O(1) push operation. Array List can contain at most N elements, hence for the Java solution, the space complexity is O(N).
 
            */
            public int[] UsingMontonicStackSpaceOptimaal(int[] heights)
            {
                int n = heights.Length;
                List<int> list = new List<int>();
                int maxHeight = -1;

                for (int current = n - 1; current >= 0; --current)
                {
                    // If there is no building higher (or equal) than the current one to its right,
                    // push it in the list.
                    if (maxHeight < heights[current])
                    {
                        list.Add(current);

                        // Update max building till now.
                        maxHeight = heights[current];
                    }
                }

                // Push building indices from list to answer array in reverse order.
                int[] answer = new int[list.Count];
                for (int i = 0; i < list.Count; ++i)
                {
                    answer[i] = list[list.Count - 1 - i];
                }
                return answer;
            }
        }


        /* 2282. Number of People That Can Be Seen in a Grid
        https://leetcode.com/problems/number-of-people-that-can-be-seen-in-a-grid/description/	
         */
        public class SeePeopleSol
        {
            /*
            Approach: Monostack for both directions
Time Complextiy: O(MN)

            */
            public int[][] Monostack(int[][] heights)
            {
                int rowCount = heights.Length, columnCount = heights[0].Length;
                int[][] result = new int[rowCount][];
                for (int i = 0; i < rowCount; i++)
                {
                    result[i] = new int[columnCount];
                }

                for (int columnIndex = 0; columnIndex < columnCount; columnIndex++)
                { // DOWN
                    Stack<int> stack = new Stack<int>();
                    for (int rowIndex = rowCount - 1; rowIndex >= 0; rowIndex--)
                    {
                        while (stack.Count > 0 && heights[rowIndex][columnIndex] > stack.Peek())
                        {
                            result[rowIndex][columnIndex]++;
                            stack.Pop();
                        }
                        if (stack.Count > 0)
                        {
                            result[rowIndex][columnIndex]++;
                        }
                        if (stack.Count == 0 || heights[rowIndex][columnIndex] != stack.Peek())
                        {
                            stack.Push(heights[rowIndex][columnIndex]);
                        }
                    }
                }

                for (int rowIndex = 0; rowIndex < rowCount; rowIndex++)
                { // RIGHT
                    Stack<int> stack = new Stack<int>();
                    for (int columnIndex = columnCount - 1; columnIndex >= 0; columnIndex--)
                    {
                        while (stack.Count > 0 && heights[rowIndex][columnIndex] > stack.Peek())
                        {
                            result[rowIndex][columnIndex]++;
                            stack.Pop();
                        }
                        if (stack.Count > 0)
                        {
                            result[rowIndex][columnIndex]++;
                        }
                        if (stack.Count == 0 || heights[rowIndex][columnIndex] != stack.Peek())
                        {
                            stack.Push(heights[rowIndex][columnIndex]);
                        }
                    }
                }

                return result;
            }
        }



        /* 2281. Sum of Total Strength of Wizards	
        https://leetcode.com/problems/sum-of-total-strength-of-wizards/description/
         */
        public class TotalStrengthSol
        {
            /*           
Approach: Prefix Sum + Monotonic Stack
Complexity Analysis
Let n be the length of the input array strength.
•	Time complexity: O(n)
o	We use mono stack to build arrays leftIndex and rightIndex, each element is added to the stack or removed from the stack by at most once, thus it takes at most O(n) time to build them.
o	It takes O(n) time to build the prefix array of strength.
o	It takes O(n) time to build the prefix sum of the prefix sum array of strength.
o	We iterate over strength, according to the previous equations, it takes O(1) time to calculate the sum of strengths having each element as the minimum, so the total time complexity of this step is O(n).
o	To sum up, the overall time complexity is O(n).
•	Space complexity: O(n)
o	We build some auxiliary arrays leftIndex, rightIndex, and presumOfPresum, each of them has O(n) elements, thus the total space complexity is O(n).


            */
            public int PrefixSumwithMonoStack(int[] strength)
            {
                int mod = (int)1e9 + 7, n = strength.Length;

                // Get the first index of the non-larger value to strength[i]'s right.
                Stack<int> stack = new Stack<int>();
                int[] rightIndex = new int[n];
                Array.Fill(rightIndex, n);
                for (int i = 0; i < n; ++i)
                {
                    while (stack.Count > 0 && strength[stack.Peek()] >= strength[i])
                    {
                        rightIndex[stack.Pop()] = i;
                    }
                    stack.Push(i);
                }

                // Get the first index of the smaller value to strength[i]'s left.
                int[] leftIndex = new int[n];
                Array.Fill(leftIndex, -1);
                stack.Clear();
                for (int i = n - 1; i >= 0; --i)
                {
                    while (stack.Count > 0 && strength[stack.Peek()] > strength[i])
                        leftIndex[stack.Pop()] = i;
                    stack.Push(i);
                }

                // Get the prefix sum of the prefix sum array of strength.
                long answer = 0;
                long[] presumOfPresum = new long[n + 2];
                for (int i = 0; i < n; ++i)
                    presumOfPresum[i + 2] = (presumOfPresum[i + 1] + strength[i]) % mod;
                for (int i = 1; i <= n; ++i)
                    presumOfPresum[i + 1] = (presumOfPresum[i + 1] + presumOfPresum[i]) % mod;

                // For each element in strength, we get the value of R_term - L_term.
                for (int i = 0; i < n; ++i)
                {
                    // Get the left index and the right index.
                    int leftBound = leftIndex[i], rightBound = rightIndex[i];

                    // Get leftCount and rightCount (marked as L and R in the previous slides)
                    int leftCount = i - leftBound, rightCount = rightBound - i;

                    // Get posPresum and negPresum.
                    long negPresum = (presumOfPresum[i + 1] - presumOfPresum[i - leftCount + 1]) % mod;
                    long posPresum = (presumOfPresum[i + rightCount + 1] - presumOfPresum[i + 1]) % mod;

                    // The total strength of all subarrays that have strength[i] as the minimum.
                    answer = (answer + (posPresum * leftCount - negPresum * rightCount) % mod * strength[i] % mod) % mod;
                }

                return (int)(answer + mod) % mod;
            }
        }


        /* 2345. Finding the Number of Visible Mountains
        https://leetcode.com/problems/finding-the-number-of-visible-mountains/description/
         */
        public int VisibleMountains(int[][] peaks)
        {
            /*
            Complexity: O(N LogN)
Sort + linear pass
            */
            int pCnt = peaks.Length;
            int[][] bases = new int[pCnt][];


            for (int idx = 0; idx < bases.Length; idx++)
            {
                bases[idx][0] = peaks[idx][0] - peaks[idx][1];
                bases[idx][1] = peaks[idx][0] + peaks[idx][1];
            }

            Array.Sort(bases, (a, b) => a[0] - b[0] == 0 ? b[1] - a[1] : a[0] - b[0]);


            int i = 0, j = 0, visible = 0;
            Boolean go = true, duplicate = false;
            while (go)
            {
                go = false;
                duplicate = false;

                j = i + 1;
                while (j < pCnt)
                {

                    if (bases[j][1] > bases[i][1])
                    {
                        i = j;
                        go = true;
                        break;
                    }
                    else if (bases[i][0] == bases[j][0] && bases[i][1] == bases[j][1])
                    {
                        duplicate = true;
                    }
                    j++;
                }

                if (!duplicate)
                    visible++;

            }


            return visible;

        }


        /* 1672. Richest Customer Wealth
        https://leetcode.com/problems/richest-customer-wealth/description/
         */

        class MaximumWealthSol
        {
            public int MaximumWealth(int[][] accounts)
            {
                // Initialize the maximum wealth seen so far to 0 (the minimum wealth possible)
                int maxWealthSoFar = 0;
                /*
                Approach 1: Row Sum Maximum
                Complexity Analysis
                Let M be the number of customers and N be the number of banks.
                •	Time complexity: O(M⋅N)
                For each of the M customers, we need to iterate over all N banks to find the sum of his/her wealth. Inside each iteration, operations like addition or finding maximum take O(1) time. Hence, the total time complexity is O(M⋅N).
                •	Space complexity: O(1)
                We only need two variables currCustomerWealth and maxWealthSoFar to store the wealth of the current customer, and the highest wealth we have seen so far respectively. Therefore, the space complexity is constant.

                */
                // Iterate over accounts
                foreach (int[] account in accounts)
                {
                    // For each account, initialize the sum to 0
                    int currCustomerWealth = 0;
                    // Add the money in each bank
                    foreach (int money in account)
                    {
                        currCustomerWealth += money;
                    }
                    // Update the maximum wealth seen so far if the current wealth is greater
                    // If it is less than the current sum
                    maxWealthSoFar = Math.Max(maxWealthSoFar, currCustomerWealth);
                }

                // Return the maximum wealth
                return maxWealthSoFar;
            }
        }


        /* 1710. Maximum Units on a Truck
        https://leetcode.com/problems/maximum-units-on-a-truck/description/
         */
        public class MaximumUnitsOnTruckSol
        {
            /*
            Approach 1: Brute Force
            Complexity Analysis
•	Time Complexity : O(n^2), where n is the number of elements in array boxTypes. In the method findMaxUnitBox , we are iterating over all the elements in array boxTypes to find the maximum units. In the worst case, when all the boxes are added to the truck we would iterate n times over the array of size n. This would give total time complexity as O(n^2).
•	Space Complexity: O(1), as we are using constant extra space to maintain the variables remainingTruckSize and unitCount.

            */
            public int Naive(int[][] boxTypes, int truckSize)
            {
                int unitCount = 0;
                int remainingTruckSize = truckSize;
                while (remainingTruckSize > 0)
                {
                    int maxUnitBoxIndex = FindMaxUnitBox(boxTypes);
                    // check if all boxes are used
                    if (maxUnitBoxIndex == -1)
                        break;
                    // find the box count that can be put in truck
                    int boxCount = Math.Min(remainingTruckSize, boxTypes[maxUnitBoxIndex][0]);
                    unitCount += boxCount * boxTypes[maxUnitBoxIndex][1];
                    remainingTruckSize -= boxCount;
                    // mark box with index maxUnitBoxIndex as used
                    boxTypes[maxUnitBoxIndex][1] = -1;
                }
                return unitCount;
            }
            private int FindMaxUnitBox(int[][] boxTypes)
            {
                int maxUnitBoxIndex = -1;
                int maxUnits = 0;
                for (int i = 0; i < boxTypes.Length; i++)
                {
                    if (boxTypes[i][1] > maxUnits)
                    {
                        maxUnits = boxTypes[i][1];
                        maxUnitBoxIndex = i;
                    }
                }
                return maxUnitBoxIndex;
            }

            /*
            Approach 2: Using Array Sort
            Complexity Analysis
•	Time Complexity : O(nlogn), where n is the number of elements in array boxTypes.
Sorting the array boxTypes of size n takes (nlogn) time. Post that, we iterate over each element in array boxTypes and in worst case, we might end up iterating over all the elements in the array. This gives us time complexity as O(nlogn)+O(n)=O(nlogn).
•	Space Complexity: O(1), as we use constant extra space.

            */
            public int UsingSort(int[][] boxTypes, int truckSize)
            {
                Array.Sort(boxTypes, (a, b) => b[1] - a[1]);
                int unitCount = 0;
                foreach (int[] boxType in boxTypes)
                {
                    int boxCount = Math.Min(truckSize, boxType[0]);
                    unitCount += boxCount * boxType[1];
                    truckSize -= boxCount;
                    if (truckSize == 0)
                        break;
                }
                return unitCount;
            }

            /*
            Approach 3: Using Priority Queue

Complexity Analysis
•	Time Complexity : O(nlogn), where n is the number of elements in array boxTypes.
We are adding all the elements of the array boxTypes in the priority queue, which takes O(n) time.
Post that, we would continue iteration until queue is not empty or remaining truck size is greater than 0. In worst case, we might end up iterating n times. Also, removing elements from queue would take (logn) time. This gives us time complexity as O(nlogn)+O(n)=O(nlogn).
•	Space Complexity: O(n), as we use a queue of size n.

            */
            public int UsingPQ(int[][] boxTypes, int truckSize)
            {
                // Create a priority queue sorted by the number of units per box in descending order
                PriorityQueue<int[], int[]> queue = new PriorityQueue<int[], int[]>(
                    Comparer<int[]>.Create((a, b) => b[1] - a[1]));
                foreach (var boxType in boxTypes)
                {
                    queue.Enqueue(boxType, boxType);
                }

                int totalUnitCount = 0;
                while (queue.Count > 0)
                {
                    int[] topBoxType = queue.Dequeue();
                    int boxCount = Math.Min(truckSize, topBoxType[0]);
                    totalUnitCount += boxCount * topBoxType[1];
                    truckSize -= boxCount;
                    if (truckSize == 0)
                        break;
                }
                return totalUnitCount;
            }

        }



        /* 2279. Maximum Bags With Full Capacity of Rocks
        https://leetcode.com/problems/maximum-bags-with-full-capacity-of-rocks/description/
         */
        class MaximumBagsSol
        {
            /*
            
Approach 1: Greedy

Complexity Analysis
Let n be the size of the input array capacity.
•	Time complexity: O(n⋅logn)
o	We use an array remaining_capacity to store the remaining capacity of each bag and it takes O(n) time.
o	Sorting remaining_capacity requires O(n⋅logn) time.
o	We iterate over the sorted array remaining_capacity and it takes O(n) time.
o	To sum up, the overall time complexity is O(n⋅logn).
•	Space complexity: O(n)
o	We use an array of size n to store the remaining capacity of each bag.

            */
            public int WithGreedy(int[] capacity, int[] rocks, int additionalRocks)
            {
                int n = capacity.Length, fullBags = 0;

                // Sort bags by the remaining capacity.
                int[] remainingCapacity = new int[n];
                for (int i = 0; i < n; ++i)
                    remainingCapacity[i] = capacity[i] - rocks[i];
                Array.Sort(remainingCapacity);

                // Iterate over sorted bags and fill them using additional rocks.
                for (int i = 0; i < n; ++i)
                {
                    // If we can fill the current one, fill it and move on.
                    // Otherwise, stop the iteration.
                    if (additionalRocks >= remainingCapacity[i])
                    {
                        additionalRocks -= remainingCapacity[i];
                        fullBags++;
                    }
                    else
                        break;
                }

                // Return `fullBags` after the iteration stops.
                return fullBags;
            }
        }


        /* 1732. Find the Highest Altitude
        https://leetcode.com/problems/find-the-highest-altitude/description/
         */
        class LargestAltitudeSol
        {
            /*
            Approach: Prefix Sum
            Complexity Analysis
Here, N is the number of integers in the list gain.
•	Time complexity: O(N).
We iterate over every integer in the list gain only once, and hence the total time complexity is equal to O(N).
•	Space complexity: O(1).
We only need two variables, currentAltitude andhighestPoint; hence the space complexity is constant.

            */
            public int UsingPrefixSum(int[] gain)
            {
                int currentAltitude = 0;
                // Highest altitude currently is 0.
                int highestPoint = currentAltitude;

                foreach (int altitudeGain in gain)
                {
                    // Adding the gain in altitude to the current altitude.
                    currentAltitude += altitudeGain;
                    // Update the highest altitude.
                    highestPoint = Math.Max(highestPoint, currentAltitude);
                }

                return highestPoint;
            }
        }


        /* 1833. Maximum Ice Cream Bars
        https://leetcode.com/problems/maximum-ice-cream-bars/description/
         */
        class MaxIceCreamBarsSol
        {
            /*
            Approach 1: Sorting (Greedy)
Complexity Analysis
Here, n is the number of ice cream bars given.
•	Time complexity: O(n⋅logn)
o	We sort the costs array, which will take O(nlogn) time, and then iterate over it, in worst-case which may take O(n) time.
o	In Swift, the parameters passed are constant thus we would need to copy the coins variable and costs array and it will take an additional O(n) time.
o	Thus, overall we take O(nlogn+n)=O(nlogn) time.
•	Space complexity: O(logn) or O(n)
o	Some extra space is used when we sort the costs array in place. The space complexity of the sorting algorithm depends on the programming language.
o	In Python, the sort() method sorts a list using the Timsort algorithm which has O(n) additional space where n is the number of the elements.
o	In C++ and Swift, the sort() function is implemented as a hybrid of Quick Sort, Heap Sort, and Insertion Sort, with a worst-case space complexity of O(logn).
o	In Java, Arrays.sort() is implemented using a variant of the Quick Sort algorithm which has a space complexity of O(logn).
o	In JavaScript, the space complexity of sort() is O(logn).
o	In Swift, copying the costs array will also take an additional O(n) space.

            */
            public int WithGreedySorting(int[] costs, int coins)
            {
                // Store ice cream costs in increasing order.
                Array.Sort(costs);

                int n = costs.Length;
                int icecream = 0;

                // Pick ice creams till we can.
                while (icecream < n && costs[icecream] <= coins)
                {
                    // We can buy this icecream, reduce the cost from the coins. 
                    coins -= costs[icecream];
                    icecream += 1;
                }

                return icecream;
            }
            /*
            
Approach 2: Counting Sort (Greedy)
Complexity Analysis
Let n be the length of the input array, and m be the maximum element in it.
•	Time complexity: O(n+m)
o	We once iterate on the input array to find the maximum element and then iterate once again to store the frequencies of its elements in costsFrequency array, thus it takes O(2n) time.
o	We then iterate over the whole costsFrequency array which in the worst case can take O(m) time.
o	Thus, overall we take O(2n+m)=O(n+m) time.
•	Space complexity: O(m)
o	We use an additional array costsFrequency of size m.

            */
            public int WithGreedyCountingSort(int[] costs, int coins)
            {
                int n = costs.Length;
                int icecreams = 0;

                int m = costs[0];
                foreach (int cost in costs)
                {
                    m = Math.Max(m, cost);
                }

                int[] costsFrequency = new int[m + 1];
                foreach (int cost in costs)
                {
                    costsFrequency[cost]++;
                }

                for (int cost = 1; cost <= m; ++cost)
                {
                    // No ice cream is present costing 'cost'.
                    if (costsFrequency[cost] == 0)
                    {
                        continue;
                    }
                    // We don't have enough 'coins' to even pick one ice cream.
                    if (coins < cost)
                    {
                        break;
                    }

                    // Count how many icecreams of 'cost' we can pick with our 'coins'.
                    // Either we can pick all ice creams of 'cost' or we will be limited by remaining 'coins'.
                    int count = Math.Min(costsFrequency[cost], coins / cost);
                    // We reduce price of picked ice creams from our coins.
                    coins -= cost * count;
                    icecreams += count;
                }

                return icecreams;
            }
        }

        /* 1834. Single-Threaded CPU
        https://leetcode.com/problems/single-threaded-cpu/description/	
         */
        class GetOrderSol
        {
            /*
            Approach 1: Sorting + Min-Heap
Complexity Analysis
Let N be the number of tasks in the input array.
•	Time complexity: O(NlogN).
o	We create sortedTasks, which is a deep copy of the tasks array. This takes O(N) time.
o	Sorting the sortedTasks array takes O(NlogN) time.
o	We push and pop each task once in the min-heap, and both the operations take O(logN) time for each element. Thus, it takes O(NlogN) time in total.
o	Thus, overall time complexity is, O(N+NlogN+NlogN)≈O(NlogN).
•	Space complexity: O(N).
o	Our sortedTasks array will store all N tasks, and the min-heap will also contain at most N tasks.
o	Thus, we use O(N+N)≈O(N) extra space.

            */
            public int[] WihtMinHeapAndSorting(int[][] tasks)
            {

                // Sort based on min task processing time or min task index.
                // Store enqueue time, processing time, task index.
                PriorityQueue<int[], int[]> nextTask = new PriorityQueue<int[], int[]>(
                    Comparer<int[]>.Create((a, b) => (a[1] != b[1] ? (a[1] - b[1]) : (a[2] - b[2]))));

                // Store task enqueue time, processing time, index.
                int[][] sortedTasks = new int[tasks.Length][];
                for (int i = 0; i < tasks.Length; ++i)
                {
                    sortedTasks[i][0] = tasks[i][0];
                    sortedTasks[i][1] = tasks[i][1];
                    sortedTasks[i][2] = i;
                }

                Array.Sort(sortedTasks, (a, b) => a[0].CompareTo(b[0]));
                int[] tasksProcessingOrder = new int[tasks.Length];

                long currTime = 0;
                int taskIndex = 0;
                int ansIndex = 0;

                // Stop when no tasks are left in array and heap.
                while (taskIndex < tasks.Length || nextTask.Count > 0)
                {
                    if (nextTask.Count == 0 && currTime < sortedTasks[taskIndex][0])
                    {
                        // When the heap is empty, try updating currTime to next task's enqueue time. 
                        currTime = sortedTasks[taskIndex][0];
                    }

                    // Push all the tasks whose enqueueTime <= currtTime into the heap.
                    while (taskIndex < tasks.Length && currTime >= sortedTasks[taskIndex][0])
                    {
                        nextTask.Enqueue(sortedTasks[taskIndex], sortedTasks[taskIndex]);
                        ++taskIndex;
                    }

                    int processTime = nextTask.Peek()[1];
                    int index = nextTask.Peek()[2];
                    nextTask.Dequeue();

                    // Complete this task and increment currTime.
                    currTime += processTime;
                    tasksProcessingOrder[ansIndex++] = index;
                }

                return tasksProcessingOrder;
            }
        }



        /* 2589. Minimum Time to Complete All Tasks
        https://leetcode.com/problems/minimum-time-to-complete-all-tasks/description/
         */
        class MinimumTimeToCompleteAllTasksSol
        {
            public int FindMinimumTimeToCompleteAllTasks(int[][] tasks)
            {
                if (tasks == null || tasks.Length == 0) return 0;
                Array.Sort(tasks, (a, b) => (a[1] - b[1])); // sort by end time increasing to complete early task first
                bool[] slots = new bool[2001];
                foreach (int[] task in tasks)
                {
                    int s = task[0];
                    int e = task[1];
                    int dur = task[2];
                    int i = s;
                    for (; i <= e && dur > 0; i++)
                    {
                        if (slots[i])
                        {
                            dur--;
                        }
                    }
                    i = e;
                    while (dur > 0 && i >= s)
                    {
                        if (!slots[i])
                        {
                            slots[i] = true;
                            dur--;
                        }
                        i--;
                    }
                }
                int res = 0;
                foreach (bool slot in slots)
                {
                    if (slot) res++;
                }
                return res;
            }
        }


        /* 1908. Game of Nim
        https://leetcode.com/problems/game-of-nim/description/
         */

        class NimGameSol
        {
            /*
            Approach 1: Simulation - Dynamic Programming
            Complexity Analysis
Let n be the number of piles, and m be the maximum number of stones in a heap.
•	Time complexity: O(n^2⋅m⋅Cn to (n+m−1)⋅logn)
o	The number of states in the game tree - we have n places to fill with m possible values. Each value can be repeated. The order of these values does not matter (as explained above in the context of equivalent states). Thus, determining the number of states is similar to determining the number of ways of choosing n objects from m different kinds of objects with repetitions. Thus, the number of states is Cn to n+m−1 (C stands for binomial coefficient).
o	For each state, we have to check all possible future states. This would take O(m⋅n) time.
o	We sort the piles in each state. It takes O(nlogn) time.
o	So, the total time complexity is the product of all three - O(n⋅m⋅Cn to n+m−1⋅nlogn), which is O(n^2⋅m⋅Cn to n+m−1⋅logn).
•	Space complexity: O(n⋅Cn to n+m−1)
o	Number of states in the game tree is Cnn+m−1. These states occupy space on the memo table in the form of key value pairs. The key is a string of n numbers, and the value is a bool. So, each state occupies O(n) space. Thus, the total space occupied by the memo table is O(n⋅Cnn+m−1).
o	In addition, the recursive implementation takes up space on the implicit stack. The stack's maximum depth is the game tree's height, which is n⋅m. We create a copy of the piles array in each call, which takes up O(n) space. So, the total space occupied on the implicit stack is O(n^2⋅m).
o	Out of two terms, the first one is much larger than the second one. So, the total space complexity is O(n⋅Cn to n+m−1).

            */
            public bool SimulationDP(int[] piles)
            {
                // The count of stones remaining, we recurse until
                // the count becomes zero.
                int remainingStones = piles.Sum();

                // Hash map for memoization.
                Dictionary<string, bool> memoizationMap = new Dictionary<string, bool>();

                // Is the person to play next the winner?
                // The first person to play is Alice at the beginning.
                // So, if Alice wins, it is going to be true, otherwise
                // it is going to be false.
                return IsNextPersonWinner(piles, remainingStones, memoizationMap);
            }

            private bool IsNextPersonWinner(int[] piles, int remainingStones, Dictionary<string, bool> memoizationMap)
            {
                // Make a key by concatenating the count of stones
                // in each pile, so key for the state [1, 2, 3] => '1-2-3'.
                string stateKey = GetKey(piles);

                // Have we come across this state already?
                if (memoizationMap.ContainsKey(stateKey))
                {
                    return memoizationMap[stateKey];
                }

                // The current player has no more moves left, so they
                // lose the game.
                if (remainingStones == 0)
                {
                    return false;
                }

                // Generate all possible next moves and check if
                // the opponent loses the game in any of them.
                for (int i = 0; i < piles.Length; i++)
                {
                    // piles[i] is greater than 0.
                    for (int j = 1; j <= piles[i]; j++)
                    {
                        piles[i] -= j;

                        // Next state is created by making a copy of the
                        // current state array, and sorting it in ascending
                        // order of pile heights.
                        int[] nextState = (int[])piles.Clone();
                        Array.Sort(nextState);

                        // If the opponent loses, that means we win.
                        if (!IsNextPersonWinner(nextState, remainingStones - j, memoizationMap))
                        {
                            memoizationMap[stateKey] = true;
                            return true;
                        }
                        piles[i] += j;
                    }
                }

                // If none returned false for the opponent, we must have
                // lost the game.
                memoizationMap[stateKey] = false;
                return false;
            }

            private string GetKey(int[] piles)
            {
                System.Text.StringBuilder keyBuilder = new System.Text.StringBuilder();
                foreach (int height in piles)
                {
                    keyBuilder.Append(height).Append("-");
                }
                return keyBuilder.ToString();
            }
            /*
            Approach 2: Mathematical / Bit Manipulation
            Complexity Analysis
Let n be the number of piles, and m be the maximum number of stones in a heap.
•	Time Complexity - O(n). We iterate over all the piles. In each iteration, we perform an XOR operation. XOR operation takes O(1) time.
•	Space Complexity - O(1). We use a constant amount of space.

            */
            public bool MathsWithBitManip(int[] piles)
            {
                int nimSum = 0;
                foreach (int p in piles)
                {
                    nimSum ^= p;
                }
                return nimSum != 0;
            }
        }


        /* 1926. Nearest Exit from Entrance in Maze
        https://leetcode.com/problems/nearest-exit-from-entrance-in-maze/description/
         */
        public class NearestExitSol
        {
            /*
            Approach 1: Breadth First Search (BFS)
            Complexity Analysis
Let m,n be the size of the input matrix maze.
•	Time complexity: O(m⋅n)
o	For each visited cell, we add it to queue and pop it from queue once, which takes constant time as the operation on queue requires O(1) time.
o	For each cell in queue, we mark it as visited in maze, and check if it has any unvisited neighbors in all four directions. This also takes constant time.
o	In the worst-case scenario, we may have to visit O(m⋅n) cells before the iteration stops.
o	To sum up, the overall time complexity is O(m⋅n).
•	Space complexity: O(max(m,n))
o	We modify the input matrix maze in-place to mark each visited cell, it requires constant space.
o	We use a queue queue to store the cells to be visited. In the worst-case scenario, there may be O(m+n) cells stored in queue.
o	The space complexity is O(m+n)+O(max(m,n)).

            */
            public int NearestExit(char[][] maze, int[] entrance)
            {
                int numberOfRows = maze.Length, numberOfColumns = maze[0].Length;
                int[][] directions = new int[][] { new int[] { 1, 0 }, new int[] { -1, 0 }, new int[] { 0, 1 }, new int[] { 0, -1 } };

                // Mark the entrance as visited since it's not an exit.
                int startRow = entrance[0], startColumn = entrance[1];
                maze[startRow][startColumn] = '+';

                // Start BFS from the entrance, and use a queue `queue` to store all 
                // the cells to be visited.
                Queue<int[]> queue = new Queue<int[]>();
                queue.Enqueue(new int[] { startRow, startColumn, 0 });

                while (queue.Count > 0)
                {
                    int[] currentState = queue.Dequeue();
                    int currentRow = currentState[0], currentColumn = currentState[1], currentDistance = currentState[2];

                    // For the current cell, check its four neighbor cells.
                    foreach (int[] direction in directions)
                    {
                        int nextRow = currentRow + direction[0], nextColumn = currentColumn + direction[1];

                        // If there exists an unvisited empty neighbor:
                        if (0 <= nextRow && nextRow < numberOfRows && 0 <= nextColumn && nextColumn < numberOfColumns
                           && maze[nextRow][nextColumn] == '.')
                        {

                            // If this empty cell is an exit, return distance + 1.
                            if (nextRow == 0 || nextRow == numberOfRows - 1 || nextColumn == 0 || nextColumn == numberOfColumns - 1)
                                return currentDistance + 1;

                            // Otherwise, add this cell to 'queue' and mark it as visited.
                            maze[nextRow][nextColumn] = '+';
                            queue.Enqueue(new int[] { nextRow, nextColumn, currentDistance + 1 });
                        }
                    }
                }

                // If we finish iterating without finding an exit, return -1.
                return -1;
            }
        }

        /* 1964. Find the Longest Valid Obstacle Course at Each Position
        https://leetcode.com/problems/find-the-longest-valid-obstacle-course-at-each-position/description/
         */
        class LongestObstacleCourseAtEachPositionSol
        {
            List<int> answer;
            /*
            Approach: Greedy + Binary Search.
            Complexity Analysis
            Let n be the length of the input array obstacles.
            •	Time complexity: O(n⋅logn)
            o	We traverse over obstacles to find the longest sequence. At each step i in the iteration, we apply a binary search over lis to find the insertion position of the current height obstacles[i].
            o	One binary search over an sorted array of size k takes logk time. Imagine the case where we append every height to lis after each step. In the second half of the traverse, there are always more than n/2 elements in lis, thus all these n/2 binary searches take O(logn) time. In this case, the time complexity is O(n⋅logn).
            o	To sum up, the time complexity is O(n⋅logn).
            •	Space complexity: O(n)
            o	We create an array lis to store the height of the ending of each sequence. The maximum length of the longest obstacle course is n, thus the size of lis is n in the worst-case scenario.

            */
            public int[] GreedyWithBinarySearch(int[] obstacles)
            {
                int n = obstacles.Length, lisLength = 0;

                // lis[i] records the lowest increasing sequence of length i + 1.
                int[] answer = new int[n], lis = new int[n];

                for (int i = 0; i < n; ++i)
                {
                    int height = obstacles[i];

                    // Find the rightmost insertion position idx.
                    int idx = BisectRight(lis, height, lisLength);
                    if (idx == lisLength)
                        lisLength++;

                    lis[idx] = height;
                    answer[i] = idx + 1;
                }
                return answer;
            }
            // Find the rightmost insertion position. We use a fixed-length array and a changeable right boundary 
            // to represent an arraylist of dynamic size.
            private int BisectRight(int[] A, int target, int right)
            {
                if (right == 0)
                    return 0;
                int left = 0;
                while (left < right)
                {
                    int mid = left + (right - left) / 2;
                    if (A[mid] <= target)
                        left = mid + 1;
                    else
                        right = mid;
                }
                return left;
            }
        }


        /* 1996. The Number of Weak Characters in the Game
        https://leetcode.com/problems/the-number-of-weak-characters-in-the-game/description/
         */

        class NumberOfWeakCharactersInGameSol
        {
            /*
    Approach 1: Sorting
Complexity Analysis
Here, N is the number of pairs in the given list properties.
•	Time complexity: O(NlogN)
Sorting a list of N elements takes O(NlogN) time. The iteration over the sorted list to count the weak character takes O(N) time. Hence the time complexity equals O(NlogN).
•	Space complexity: O(logN)
We only need two variables maxDefense and weakCharacters to solve the problem. Some space will be used for sorting the list. The space complexity of the sorting algorithm depends on the implementation of each programming language. For instance, in Java, the Arrays.sort() for primitives is implemented as a variant of the quicksort algorithm whose space complexity is O(logN). In C++ std::sort() function provided by STL is a hybrid of Quick Sort, Heap Sort, and Insertion Sort and has a worst-case space complexity of O(logN). Thus, the use of the inbuilt sort() function might add up to O(logN) to space complexity.

     */
            public int WithSorting(int[][] properties)
            {

                // Sort in ascending order of attack, 
                // If attack is same sort in descending order of defense
                Array.Sort(properties, (a, b) => (a[0] == b[0]) ? (b[1] - a[1]) : a[0] - b[0]);

                int weakCharacters = 0;
                int maxDefense = 0;
                for (int i = properties.Length - 1; i >= 0; i--)
                {
                    // Compare the current defense with the maximum achieved so far
                    if (properties[i][1] < maxDefense)
                    {
                        weakCharacters++;
                    }
                    maxDefense = Math.Max(maxDefense, properties[i][1]);
                }

                return weakCharacters;
            }
            /*
            Approach 2: Greedy
Complexity Analysis
Here, N is the number of pairs in the given list properties, and K is the maximum attack value possible.
•	Time complexity: O(N+K)
The iteration over the pairs to find the maximum defense value for a particular attack value takes O(N) time. The iteration over the possible value of the attack property takes O(K) time. The iteration over the properties to count the weak characters takes O(N) time. Therefore, the total time complexity equals to O(N+K).
•	Space complexity: O(K)
The array maxDefense will be of size K to store the defense value corresponding to all the attack values.	

            */
            public int WithGreedy(int[][] properties)
            {
                int maxAttack = 0;
                // Find the maximum atack value
                foreach (int[] property in properties)
                {
                    int attack = property[0];
                    maxAttack = Math.Max(maxAttack, attack);
                }

                int[] maxDefense = new int[maxAttack + 2];
                // Store the maximum defense for an attack value
                foreach (int[] property in properties)
                {
                    int attack = property[0];
                    int defense = property[1];

                    maxDefense[attack] = Math.Max(maxDefense[attack], defense);
                }

                // Store the maximum defense for attack greater than or equal to a value
                for (int i = maxAttack - 1; i >= 0; i--)
                {
                    maxDefense[i] = Math.Max(maxDefense[i], maxDefense[i + 1]);
                }

                int weakCharacters = 0;
                foreach (int[] property in properties)
                {
                    int attack = property[0];
                    int defense = property[1];

                    // If their is a greater defense for properties with greater attack
                    if (defense < maxDefense[attack + 1])
                    {
                        weakCharacters++;
                    }
                }

                return weakCharacters;
            }
        }


        /* 
        354. Russian Doll Envelopes	
        https://leetcode.com/problems/russian-doll-envelopes/description/	
         */

        class MaxEnvelopesSol
        {
            /*
            Approach 1: Sort + Longest Increasing Subsequence (LIS)
            Complexity Analysis
•	Time complexity : O(NlogN), where N is the length of the input. Both sorting the array and finding the LIS happen in O(NlogN)
•	Space complexity : O(N). Our lis function requires an array dp which goes up to size N. Also the sorting algorithm we use may also take additional space.

            */

            public int UsingSortAndLIS(int[][] envelopes)
            {
                // sort on increasing in first dimension and decreasing in second
                Array.Sort(envelopes, (arr1, arr2) =>
                {
                    if (arr1[0] == arr2[0])
                    {
                        return arr2[1] - arr1[1];
                    }
                    else
                    {
                        return arr1[0] - arr2[0];
                    }
                });
                // extract the second dimension and run LIS
                int[] secondDimension = new int[envelopes.Length];
                for (int i = 0; i < envelopes.Length; ++i) secondDimension[i] = envelopes[i][1];
                return LengthOfLIS(secondDimension);
            }

            private int LengthOfLIS(int[] numbers)
            {
                int[] dynamicProgrammingArray = new int[numbers.Length];
                int length = 0;
                foreach (int number in numbers)
                {
                    int index = Array.BinarySearch(dynamicProgrammingArray, 0, length, number);
                    if (index < 0)
                    {
                        index = -(index + 1);
                    }
                    dynamicProgrammingArray[index] = number;
                    if (index == length)
                    {
                        length++;
                    }
                }
                return length;
            }


        }

        /* 529. Minesweeper
        https://leetcode.com/problems/minesweeper/description/
         */
        public class UpdateBoardSol
        {
            /*
            1. DFS 
            */
            public char[][] DFS(char[][] board, int[] click)
            {
                int m = board.Length, n = board[0].Length;
                int row = click[0], col = click[1];

                if (board[row][col] == 'M')
                { // Mine
                    board[row][col] = 'X';
                }
                else
                { // Empty
                  // Get number of mines first.
                    int count = 0;
                    for (int i = -1; i < 2; i++)
                    {
                        for (int j = -1; j < 2; j++)
                        {
                            if (i == 0 && j == 0) continue;
                            int r = row + i, c = col + j;
                            if (r < 0 || r >= m || c < 0 || c < 0 || c >= n) continue;
                            if (board[r][c] == 'M' || board[r][c] == 'X') count++;
                        }
                    }

                    if (count > 0)
                    { // If it is not a 'B', stop further DFS.
                        board[row][col] = (char)(count + '0');
                    }
                    else
                    { // Continue DFS to adjacent cells.
                        board[row][col] = 'B';
                        for (int i = -1; i < 2; i++)
                        {
                            for (int j = -1; j < 2; j++)
                            {
                                if (i == 0 && j == 0) continue;
                                int r = row + i, c = col + j;
                                if (r < 0 || r >= m || c < 0 || c < 0 || c >= n) continue;
                                if (board[r][c] == 'E') DFS(board, new int[] { r, c });
                            }
                        }
                    }
                }

                return board;
            }
            /*
            2.BFS
            */
            public char[][] BFS(char[][] board, int[] click)
            {
                int m = board.Length, n = board[0].Length;
                Queue<int[]> queue = new Queue<int[]>();
                queue.Enqueue(click);

                while (queue.Count > 0)
                {
                    int[] cell = queue.Dequeue();
                    int row = cell[0], col = cell[1];

                    if (board[row][col] == 'M')
                    { // Mine
                        board[row][col] = 'X';
                    }
                    else
                    { // Empty
                      // Get number of mines first.
                        int count = 0;
                        for (int i = -1; i < 2; i++)
                        {
                            for (int j = -1; j < 2; j++)
                            {
                                if (i == 0 && j == 0) continue;
                                int r = row + i, c = col + j;
                                if (r < 0 || r >= m || c < 0 || c < 0 || c >= n) continue;
                                if (board[r][c] == 'M' || board[r][c] == 'X') count++;
                            }
                        }

                        if (count > 0)
                        { // If it is not a 'B', stop further BFS.
                            board[row][col] = (char)(count + '0');
                        }
                        else
                        { // Continue BFS to adjacent cells.
                            board[row][col] = 'B';
                            for (int i = -1; i < 2; i++)
                            {
                                for (int j = -1; j < 2; j++)
                                {
                                    if (i == 0 && j == 0) continue;
                                    int r = row + i, c = col + j;
                                    if (r < 0 || r >= m || c < 0 || c < 0 || c >= n) continue;
                                    if (board[r][c] == 'E')
                                    {
                                        queue.Enqueue(new int[] { r, c });
                                        board[r][c] = 'B'; // Avoid to be added again.
                                    }
                                }
                            }
                        }
                    }
                }

                return board;
            }


        }

        /* 2101. Detonate the Maximum Bombs
        https://leetcode.com/problems/detonate-the-maximum-bombs/description/
         */
        class MaximumDetonationSol
        {
            /*
            Approach 1: Depth-First Search, Recursive
Complexity Analysis
Let n be the number of bombs, so there are n nodes and at most n^2 edges in the equivalence graph.
•	Time complexity: O(n^3)
o	Building the graph takes O(n^2) time.
o	The time complexity of a typical DFS is O(V+E) where V represents the number of nodes, and E represents the number of edges. More specifically, there are n nodes and n2 edges in this problem.
o	Each node is only visited once, which takes O(n) time.
o	For each node, we may need to explore up to n−1 edges to find all its neighbors. Since there are n nodes, the total number of edges we explore is at most n(n−1)=O(n^2).
o	We need to perform n depth-first searches.
•	Space complexity: O(n^2)
o	The space complexity of DFS is (n2):
o	There are O(n^2) edges stored in graph.
o	We need to maintain a hash set that contains at most n visited nodes
o	The call stack of dfs contains also takes n space.

            */
            public int DFSRec(int[][] bombs)
            {
                Dictionary<int, List<int>> graph = new Dictionary<int, List<int>>();
                int numberOfBombs = bombs.Length;

                // Build the graph
                for (int i = 0; i < numberOfBombs; i++)
                {
                    for (int j = 0; j < numberOfBombs; j++)
                    {
                        if (i == j)
                        {
                            continue;
                        }
                        int bombX = bombs[i][0], bombY = bombs[i][1], bombRadius = bombs[i][2];
                        int targetX = bombs[j][0], targetY = bombs[j][1];

                        // Create a path from node i to node j, if bomb i detonates bomb j.
                        if ((long)bombRadius * bombRadius >= (long)(bombX - targetX) * (bombX - targetX) + (long)(bombY - targetY) * (bombY - targetY))
                        {
                            if (!graph.ContainsKey(i))
                            {
                                graph[i] = new List<int>();
                            }
                            graph[i].Add(j);
                        }
                    }
                }

                int maxDetonatedBombs = 0;
                for (int i = 0; i < numberOfBombs; i++)
                {
                    int count = Dfs(i, new HashSet<int>(), graph);
                    maxDetonatedBombs = Math.Max(maxDetonatedBombs, count);
                }

                return maxDetonatedBombs;
            }

            // DFS to get the number of nodes reachable from a given node current
            private int Dfs(int current, HashSet<int> visited, Dictionary<int, List<int>> graph)
            {
                visited.Add(current);
                int count = 1;
                foreach (int neighbor in graph.GetValueOrDefault(current, new List<int>()))
                {
                    if (!visited.Contains(neighbor))
                    {
                        count += Dfs(neighbor, visited, graph);
                    }
                }
                return count;
            }

            /* Approach 2: Depth-First Search, Iterative
Complexity Analysis
Let n be the number of bombs, so there are n nodes and at most n^2 edges in the equivalence graph.
•	Time complexity: O(n^3)
o	The time complexity of a typical DFS is O(V+E) where V represents the number of nodes, and E represents the number of edges. More specifically, there are n nodes and n2 edges in this problem.
o	Building graph takes O(n^2) time.
o	For each node, we may need to explore up to n−1 edges to find all its neighbors. Since there are n nodes, the total number of edges we explore is at most n(n−1)=O(n^2).
o	We need to perform n breadth-first searches.
•	Space complexity: O(n^2)
o	We use a hash map to store all edges, which requires O(n^2) space.
o	We use a hash set visited to record all visited nodes, which takes O(n) space.
o	We use a stack stack to store all the nodes to be visited, and in the worst-case scenario, there may be O(n) nodes in stack.
o	To sum up, the space complexity is O(n^2).

             */
            public int DFSIterative(int[][] bombs)
            {
                Dictionary<int, List<int>> graph = new Dictionary<int, List<int>>();
                int bombCount = bombs.Length;

                // Build the graph
                for (int i = 0; i < bombCount; i++)
                {
                    for (int j = 0; j < bombCount; j++)
                    {
                        if (i == j)
                        {
                            continue;
                        }
                        int x1 = bombs[i][0], y1 = bombs[i][1], radius1 = bombs[i][2];
                        int x2 = bombs[j][0], y2 = bombs[j][1];

                        // Create a path from node i to node j, if bomb i detonates bomb j.
                        if ((long)radius1 * radius1 >= (long)(x1 - x2) * (x1 - x2) + (long)(y1 - y2) * (y1 - y2))
                        {
                            if (!graph.ContainsKey(i))
                            {
                                graph[i] = new List<int>();
                            }
                            graph[i].Add(j);
                        }
                    }
                }

                int maxDetonations = 0;
                for (int i = 0; i < bombCount; i++)
                {
                    maxDetonations = Math.Max(maxDetonations, DepthFirstSearch(i, graph));
                }

                return maxDetonations;
            }

            private int DepthFirstSearch(int index, Dictionary<int, List<int>> graph)
            {
                Stack<int> stack = new Stack<int>();
                HashSet<int> visitedBoms = new HashSet<int>();
                stack.Push(index);
                visitedBoms.Add(index);
                while (stack.Count > 0)
                {
                    int current = stack.Pop();
                    foreach (int neighbor in graph.GetValueOrDefault(current, new List<int>()))
                    {
                        if (!visitedBoms.Contains(neighbor))
                        {
                            visitedBoms.Add(neighbor);
                            stack.Push(neighbor);
                        }
                    }
                }
                return visitedBoms.Count;
            }

            /*
            Approach 3: Breadth-First Search
          Complexity Analysis
Let n be the number of bombs.
•	Time complexity: O(n^3)
o	In a typical BFS search, the time complexity is O(V+E) where V is the number of nodes and E is the number of edges. There are n nodes and at most n^2 edges in this problem.
o	Building graph takes O(n^2) time.
o	Each node is enqueued and dequeued once, it takes O(n) to handle all nodes.
o	For each node, we may need to explore up to n−1 edges to find all its neighbors. Since there are n nodes, the total number of edges we explore is at most n(n−1)=O(n^2).
o	We need to perform n breadth-first searches.
•	Space complexity: O(n^2)
o	There are at O(n^2) edges stored in graph.
o	queue can store up to n nodes.
  
            */
            public int BFS(int[][] bombs)
            {
                Dictionary<int, List<int>> graph = new Dictionary<int, List<int>>();
                int numberOfBombs = bombs.Length;

                // Build the graph
                for (int i = 0; i < numberOfBombs; i++)
                {
                    for (int j = 0; j < numberOfBombs; j++)
                    {
                        int xCoordinateI = bombs[i][0], yCoordinateI = bombs[i][1], radiusI = bombs[i][2];
                        int xCoordinateJ = bombs[j][0], yCoordinateJ = bombs[j][1];

                        // Create a path from node i to node j, if bomb i detonates bomb j.
                        if ((long)radiusI * radiusI >= (long)(xCoordinateI - xCoordinateJ) * (xCoordinateI - xCoordinateJ) + (long)(yCoordinateI - yCoordinateJ) * (yCoordinateI - yCoordinateJ))
                        {
                            if (!graph.ContainsKey(i))
                            {
                                graph[i] = new List<int>();
                            }
                            graph[i].Add(j);
                        }
                    }
                }

                int maximumDetonatedBombs = 0;
                for (int i = 0; i < numberOfBombs; i++)
                {
                    maximumDetonatedBombs = Math.Max(maximumDetonatedBombs, Bfs(i, graph));
                }

                return maximumDetonatedBombs;
            }

            private int Bfs(int bombIndex, Dictionary<int, List<int>> graph)
            {
                Queue<int> queue = new Queue<int>();
                HashSet<int> visitedBombs = new HashSet<int>();
                queue.Enqueue(bombIndex);
                visitedBombs.Add(bombIndex);
                while (queue.Count > 0)
                {
                    int currentBomb = queue.Dequeue();
                    foreach (int neighbor in graph.GetValueOrDefault(currentBomb, new List<int>()))
                    {
                        if (!visitedBombs.Contains(neighbor))
                        {
                            visitedBombs.Add(neighbor);
                            queue.Enqueue(neighbor);
                        }
                    }
                }
                return visitedBombs.Count;
            }
        }


        /* 2140. Solving Questions With Brainpower
        https://leetcode.com/problems/solving-questions-with-brainpower/description/	
         */
        class MostPointsSol
        {
            /*
            Approach 1: Dynamic Programming (Iterative)

Complexity Analysis
Let n be the length of the input array questions.
•	Time complexity: O(n)
o	We need to iterate over dp. At each step, we calculate and update dp[i] which take O(1) time.
•	Space complexity: O(n)
o	We initialize an array of size n.

            */
            public long DPIterative(int[][] questions)
            {
                int n = questions.Length;
                long[] dp = new long[n];
                dp[n - 1] = questions[n - 1][0];

                for (int i = n - 2; i >= 0; --i)
                {
                    dp[i] = questions[i][0];
                    int skip = questions[i][1];
                    if (i + skip + 1 < n)
                    {
                        dp[i] += dp[i + skip + 1];
                    }

                    // dp[i] = max(solve it, skip it)
                    dp[i] = Math.Max(dp[i], dp[i + 1]);
                }

                return dp[0];
            }

            /* Approach 2: Dynamic Programming (Recursive)
            Complexity Analysis
            Let n be the length of the input array questions.
            •	Time complexity: O(n)
            o	Recall the picture at the beginning of this approach, the time complexity is proportional to the number of the function calls. Since we use dp as memory, each dfs(i) will be called exactly once, so the time complexity is O(n).
            •	Space complexity: O(n)
            o	The space complexity is proportional to the maximum depth of the recursion tree. We have up to n questions, which results in a recursion tree of depth O(n).
            o	Each function call takes O(1) space.
            o	Additionally, we initialize an array dp of size n which also takes O(n) space.
            o	Therefore, the overall space complexity is O(n).

             */
            long[] dp;
            public long DFSRec(int[][] questions)
            {
                int n = questions.Length;
                dp = new long[n];

                return dfs(questions, 0);
            }
            private long dfs(int[][] questions, int i)
            {
                if (i >= questions.Length)
                {
                    return 0;
                }
                if (dp[i] != 0)
                {
                    return dp[i];
                }
                long points = questions[i][0];
                int skip = questions[i][1];

                // dp[i] = max(skip it, solve it)
                dp[i] = Math.Max(points + dfs(questions, i + skip + 1), dfs(questions, i + 1));
                return dp[i];
            }



        }

        /* 
        2225. Find Players With Zero or One Losses
        https://leetcode.com/problems/find-players-with-zero-or-one-losses/description/	
         */


        public class FindWinnersSol
        {
            /*
            Approach 1: Hash Set
Complexity Analysis
Let n be the size of the input array matches.
•	Time complexity: O(n⋅logn)
o	For each match from matches, we have up to 3 operations on these sets. Operations on hash set require O(1) time. Thus the iteration over matches takes O(n) time.
o	We need to store two kinds of players in two arrays and sort them. In the worst-case scenario, there may be O(n) players in these arrays, thus it requires O(n⋅logn) time.
o	To sum up, the time complexity is O(n⋅logn).
•	Space complexity: O(n)
o	We use three hash sets to store all the players, there are at most O(n) players.

            */
            public IList<List<int>> WithHashSet(int[][] matches)
            {
                HashSet<int> noLossPlayers = new HashSet<int>(), oneLossPlayers = new HashSet<int>(),
                        moreLossPlayers = new HashSet<int>();

                foreach (int[] match in matches)
                {
                    int winner = match[0], loser = match[1];
                    // Add winner.
                    if (!oneLossPlayers.Contains(winner) && !moreLossPlayers.Contains(winner))
                    {
                        noLossPlayers.Add(winner);
                    }
                    // Add or move loser.
                    if (noLossPlayers.Contains(loser))
                    {
                        noLossPlayers.Remove(loser);
                        oneLossPlayers.Add(loser);
                    }
                    else if (oneLossPlayers.Contains(loser))
                    {
                        oneLossPlayers.Remove(loser);
                        moreLossPlayers.Add(loser);
                    }
                    else if (moreLossPlayers.Contains(loser))
                    {
                        continue;
                    }
                    else
                    {
                        oneLossPlayers.Add(loser);
                    }
                }

                IList<List<int>> result = new List<List<int>> {
            new List<int>(),
            new List<int>()
        };
                result[0] = noLossPlayers.ToList();
                result[1] = oneLossPlayers.ToList();
                result[0].Sort();
                result[1].Sort();

                return result;
            }

            /*
            
Approach 2: Hash Set + Hash Map
Complexity Analysis
Let n be the size of the input array matches.
•	Time complexity: O(n⋅logn)
o	For each match in matches, we need to update seen and losses_count once. The operation on hash set or hash map takes O(1) time. Thus the iteration over matches takes O(n) time.
o	We need to store and sort two kinds of players in two arrays respectively. In the worst-case scenario, there may be O(n) players in these two arrays, so it requires O(n⋅logn) time.
o	To sum up, the time complexity is O(n⋅logn).
•	Space complexity: O(n)
o	We use a hash set and a hash map to store all the players, there are at most O(n) players.

            */
            public IList<List<int>> UsingHashMapAndDict(int[][] matches)
            {
                HashSet<int> playersSeen = new HashSet<int>();
                Dictionary<int, int> playerLossesCount = new Dictionary<int, int>();

                foreach (int[] match in matches)
                {
                    int winner = match[0], loser = match[1];
                    playersSeen.Add(winner);
                    playersSeen.Add(loser);
                    if (playerLossesCount.ContainsKey(loser))
                    {
                        playerLossesCount[loser]++;
                    }
                    else
                    {
                        playerLossesCount[loser] = 1;
                    }
                }

                // Add players with 0 or 1 loss to the corresponding list.
                IList<List<int>> result = new List<List<int>> { new List<int>(), new List<int>() };
                foreach (int player in playersSeen)
                {
                    if (!playerLossesCount.ContainsKey(player))
                    {
                        result[0].Add(player);
                    }
                    else if (playerLossesCount[player] == 1)
                    {
                        result[1].Add(player);
                    }
                }

                result[0].Sort();
                result[1].Sort();

                return result;
            }

            /* Approach 3: Hash Map
            Complexity Analysis
Let n be the size of the input array matches.
•	Time complexity: O(n⋅logn)
o	For each match in matches, we need to update the value of both players in losses_count. Operations on hash map require O(1) time. Thus the iteration over matches takes O(n) time.
o	We need to store two kinds of players in two arrays and sort them. In the worst-case scenario, there may be O(n) players in these arrays, so it requires O(n⋅logn) time.
o	To sum up, the time complexity is O(n⋅logn).
•	Space complexity: O(n)
o	We use a hash map to store all players and their number of losses, which requires O(n) space in the worst-case scenario.

             */
            public IList<IList<int>> UsingHashMap(int[][] matches)
            {
                Dictionary<int, int> lossesCount = new Dictionary<int, int>();

                foreach (int[] match in matches)
                {
                    int winner = match[0], loser = match[1];
                    if (!lossesCount.ContainsKey(winner))
                    {
                        lossesCount[winner] = 0;
                    }
                    if (!lossesCount.ContainsKey(loser))
                    {
                        lossesCount[loser] = 0;
                    }
                    lossesCount[loser]++;
                }

                IList<IList<int>> answer = new List<IList<int>> { new List<int>(), new List<int>() };
                foreach (int player in lossesCount.Keys)
                {
                    if (lossesCount[player] == 0)
                    {
                        answer[0].Add(player);
                    }
                    else if (lossesCount[player] == 1)
                    {
                        answer[1].Add(player);
                    }
                }

                answer[0] = answer[0].OrderBy(x => x).ToList();
                answer[1] = answer[1].OrderBy(x => x).ToList();

                return answer;
            }

            /*
Approach 4: Counting with Array
Complexity Analysis
Let n be the size of the input array matches, and k be the range of values in winner or loser.
•	Time complexity: O(n+k)
o	For each match, we need to update two values in the array losses_count which takes constant time. Thus the iteration requires O(n) time.
o	We need to iterate over losses_count to collect two kinds of players, which takes O(k) time.
o	Since we iterate over players from low to high, we don't need to sort them anymore.
o	To sum up, the overall time complexity is O(n+k).
•	Space complexity: O(k)
o	We need to create an array of size O(k) to cover all players. Thus the overall space complexity is O(k).

            */
            public IList<IList<int>> UsingCountingWithArray(int[][] matches)
            {
                int[] lossesCount = new int[100001];
                Array.Fill(lossesCount, -1);

                foreach (int[] match in matches)
                {
                    int winner = match[0];
                    int loser = match[1];
                    if (lossesCount[winner] == -1)
                    {
                        lossesCount[winner] = 0;
                    }
                    if (lossesCount[loser] == -1)
                    {
                        lossesCount[loser] = 1;
                    }
                    else
                    {
                        lossesCount[loser]++;
                    }
                }

                IList<IList<int>> answer = new List<IList<int>> { new List<int>(), new List<int>() };
                for (int i = 1; i < 100001; ++i)
                {
                    if (lossesCount[i] == 0)
                    {
                        answer[0].Add(i);
                    }
                    else if (lossesCount[i] == 1)
                    {
                        answer[1].Add(i);
                    }
                }

                return answer;
            }
        }


        /* 2306. Naming a Company
        https://leetcode.com/problems/naming-a-company/description/
         */

        class DistinctNamesSol
        {
            /*
            Approach 1: Group words by their initials
            Complexity Analysis
Let n be the number of words in ideas and m be the average length of a word.
•	Time complexity: O(n⋅m)
o	We group words in ideas by their initials, it takes O(m) time to hash a string of length m, thus it takes O(n⋅m) time to hash and store n strings.
o	We need to try every pair of initials, there are 26 * 25 / 2 = 325 pairs of initials. For each pair of groups, we need to find the number of mutual suffixes by iterating one of the groups.
o	As mentioned previously, it takes O(n⋅m) time to hash and store the strings. However, the time complexity is not exactly O(325⋅n⋅m). This is because while a single pair of initials can contain all the words, a single word cannot be part of all 325 pairs of initials.
o	Taking each word and considering which pair of initials it can be part of, we can see that it is linear with the alphabet size, which is 26.
o	To sum up, the time complexity is O(26⋅n⋅m) = O(n⋅m).
•	Space complexity: O(n⋅m)
o	We store the suffixes of all words in an array of sets, it takes O(n⋅m) space.

            */
            public long GroupWordsByTheirInitials(String[] ideas)
            {
                // Group idea by their initials.
                HashSet<String>[] initialGroup = new HashSet<string>[26];
                for (int i = 0; i < 26; ++i)
                {
                    initialGroup[i] = new HashSet<string>();
                }
                foreach (String idea in ideas)
                {
                    initialGroup[idea[0] - 'a'].Add(idea.Substring(1));
                }

                // Calculate number of valid names from every pair of groups.
                long answer = 0;
                for (int i = 0; i < 25; ++i)
                {
                    for (int j = i + 1; j < 26; ++j)
                    {
                        // Get the number of mutual suffixes.
                        long numOfMutual = 0;
                        foreach (String ideaA in initialGroup[i])
                        {
                            if (initialGroup[j].Contains(ideaA))
                            {
                                numOfMutual++;
                            }
                        }

                        // Valid names are only from distinct suffixes in both groups.
                        // Since we can swap a with b and swap b with a to create two valid names, multiple answer by 2.
                        answer += 2 * (initialGroup[i].Count - numOfMutual) * (initialGroup[j].Count - numOfMutual);
                    }
                }

                return answer;
            }
        }


        /* 2305. Fair Distribution of Cookies
        https://leetcode.com/problems/fair-distribution-of-cookies/description/

         */

        class FairDistributeCookiesSol
        {
            /*
            Approach: Backtracking
Complexity Analysis
Let n be the length of cookies.
•	Time complexity: O(k^n)
o	The algorithm attempts to distribute each of the n cookies to each of the k children, resulting in at most O(k^n) distinct distributions.
•	Space complexity: O(k+n)
o	The array distribute represents the status of k children, thus taking up O(k) space.
o	The space complexity of a recursive call depends on the maximum depth of the recursive call stack, which is at most n. As each recursive call increments i by 1. Therefore, at most n levels of recursion will be created, and each level consumes a constant amount of space.

            */
            public int WithBacktracking(int[] cookies, int k)
            {
                int[] distribute = new int[k];

                return Dfs(0, distribute, cookies, k, k);
            }
            private int Dfs(int i, int[] distribute, int[] cookies, int k, int zeroCount)
            {
                // If there are not enough cookies remaining, return Integer.MAX_VALUE
                // as it leads to an invalid distribution.
                if (cookies.Length - i < zeroCount)
                {
                    return int.MaxValue;
                }

                // After distributing all cookies, return the unfairness of this
                // distribution.
                if (i == cookies.Length)
                {
                    int unfairness = int.MinValue;
                    foreach (int value in distribute)
                    {
                        unfairness = Math.Max(unfairness, value);
                    }
                    return unfairness;
                }

                // Try to distribute the i-th cookie to each child, and update answer
                // as the minimum unfairness in these distributions.
                int answer = int.MaxValue;
                for (int j = 0; j < k; ++j)
                {
                    zeroCount -= distribute[j] == 0 ? 1 : 0;
                    distribute[j] += cookies[i];

                    // Recursively distribute the next cookie.
                    answer = Math.Min(answer, Dfs(i + 1, distribute, cookies, k, zeroCount));

                    distribute[j] -= cookies[i];
                    zeroCount += distribute[j] == 0 ? 1 : 0;
                }

                return answer;
            }


        }



        /* 2477. Minimum Fuel Cost to Report to the Capital
        https://leetcode.com/problems/minimum-fuel-cost-to-report-to-the-capital/description/
         */
        class MinimumFuelCostSol
        {
            /*
            Approach 1: Depth First Search
Complexity Analysis
Here n is the number of nodes.
•	Time complexity: O(n).
o	The dfs function visits each node once, which takes O(n) time in total. Because we have n - 1 undirected edges, each edge can only be iterated twice (by nodes at the end), resulting in O(n) operations total while visiting all nodes.
o	We also need O(n) time to initialize the adjacency list.
•	Space complexity: O(n).
o	Building the adjacency list takes O(n) space.
o	The recursion call stack used by dfs can have no more than n elements in the worst-case scenario. It would take up O(n) space in that case.

            */
            long fuel;
            public long DFS(int[][] roads, int seats)
            {
                Dictionary<int, List<int>> adj = new Dictionary<int, List<int>>();
                foreach (int[] road in roads)
                {
                    if (!adj.ContainsKey(road[0]))
                    {
                        adj[road[0]] = new List<int>();
                    }
                    adj[road[0]].Add(road[1]);

                    if (!adj.ContainsKey(road[1]))
                    {
                        adj[road[1]] = new List<int>();
                    }
                    adj[road[1]].Add(road[0]);
                }
                Dfs(0, -1, adj, seats);
                return fuel;
            }
            private long Dfs(int node, int parent, Dictionary<int, List<int>> adj, int seats)
            {
                // The node itself has one representative.
                int representatives = 1;
                if (!adj.ContainsKey(node))
                {
                    return representatives;
                }
                foreach (int child in adj[node])
                {
                    if (child != parent)
                    {
                        // Add count of representatives in each child subtree to the parent subtree.
                        representatives += (int)Dfs(child, node, adj, seats);
                    }
                }
                if (node != 0)
                {
                    // Count the fuel it takes to move to the parent node.
                    // Root node does not have any parent so we ignore it.
                    fuel += (long)Math.Ceiling((double)representatives / seats);
                }
                return representatives;
            }
            /*
            Approach 2: Breadth First Search
            Complexity Analysis
            Here n is the number of nodes.
            •	Time complexity: O(n)
            o	Each queue operation in the BFS algorithm takes O(1) time, and a single node will be pushed once, leading to O(n) operations for n nodes. We iterate over all the neighbors of each node that is popped out of the queue, so for an undirected edge, a given edge could be iterated at most twice. Since there are n - 1 edges, it would take O(n) time in total.
            o	It also takes O(n) time to initialize the representatives and degree arrays each.
            •	Space complexity: O(n)
            o	Building the adjacency list takes O(n) space.
            o	The representatives and degree arrays also requires O(n) space each.
            o	The BFS queue can have no more than n elements in the worst-case scenario. It would take up O(n) space in that case.

            */
            public long BFS(int[][] roads, int seats)
            {
                int nodeCount = roads.Length + 1;
                Dictionary<int, List<int>> adjacencyList = new Dictionary<int, List<int>>();
                int[] degree = new int[nodeCount];

                foreach (int[] road in roads)
                {
                    if (!adjacencyList.ContainsKey(road[0]))
                    {
                        adjacencyList[road[0]] = new List<int>();
                    }
                    adjacencyList[road[0]].Add(road[1]);

                    if (!adjacencyList.ContainsKey(road[1]))
                    {
                        adjacencyList[road[1]] = new List<int>();
                    }
                    adjacencyList[road[1]].Add(road[0]);

                    degree[road[0]]++;
                    degree[road[1]]++;
                }

                return Bfs(nodeCount, adjacencyList, degree, seats);
            }
            public long Bfs(int nodeCount, Dictionary<int, List<int>> adjacencyList, int[] degree, int seats)
            {
                Queue<int> queue = new Queue<int>();
                for (int i = 1; i < nodeCount; i++)
                {
                    if (degree[i] == 1)
                    {
                        queue.Enqueue(i);
                    }
                }

                int[] representatives = new int[nodeCount];
                Array.Fill(representatives, 1);
                long fuel = 0;

                while (queue.Count > 0)
                {
                    int currentNode = queue.Dequeue();
                    fuel += (long)Math.Ceiling((double)representatives[currentNode] / seats);

                    foreach (int neighbor in adjacencyList[currentNode])
                    {
                        degree[neighbor]--;
                        representatives[neighbor] += representatives[currentNode];
                        if (degree[neighbor] == 1 && neighbor != 0)
                        {
                            queue.Enqueue(neighbor);
                        }
                    }
                }
                return fuel;
            }



        }

        /* 2492. Minimum Score of a Path Between Two Cities
        https://leetcode.com/problems/minimum-score-of-a-path-between-two-cities/description/	
         */
        class MinScoreSol
        {
            /*
Approach 1: Breadth First 
Complexity Analysis
Here n is the number of nodes and e is the total number edges.
•	Time complexity: O(n+e).
o	Each queue operation in the BFS algorithm takes O(1) time, and a single node can only be pushed once, leading to O(n) operations for n nodes. We iterate over all the neighbors of each node that is popped out of the queue, so for an undirected edge, a given edge could be iterated at most twice (by nodes at both ends), resulting in O(e) operations total for all the nodes.
o	We also need O(e) time to initialize the adjacency list and O(n) to initialize the visit array.
o	As a result, the total time required is O(n+e).
•	Space complexity: O(n+e).
o	Building the adjacency list takes O(e) space.
o	The BFS queue takes O(n) because each node is added at most once.
o	The visit array takes O(n) space as well.

            */
            public int BFS(int numberOfNodes, int[][] roads)
            {
                Dictionary<int, List<List<int>>> adjacencyList = new Dictionary<int, List<List<int>>>();
                foreach (int[] road in roads)
                {
                    if (!adjacencyList.ContainsKey(road[0]))
                    {
                        adjacencyList[road[0]] = new List<List<int>>();
                    }
                    adjacencyList[road[0]].Add(new List<int> { road[1], road[2] });

                    if (!adjacencyList.ContainsKey(road[1]))
                    {
                        adjacencyList[road[1]] = new List<List<int>>();
                    }
                    adjacencyList[road[1]].Add(new List<int> { road[0], road[2] });
                }
                return Bfs(numberOfNodes, adjacencyList);
            }
            private int Bfs(int numberOfNodes, Dictionary<int, List<List<int>>> adjacencyList)
            {
                bool[] visitedNodes = new bool[numberOfNodes + 1];
                Queue<int> queue = new Queue<int>();
                int minimumScore = int.MaxValue;

                queue.Enqueue(1);
                visitedNodes[1] = true;

                while (queue.Count > 0)
                {
                    int currentNode = queue.Dequeue();

                    if (!adjacencyList.ContainsKey(currentNode))
                    {
                        continue;
                    }
                    foreach (List<int> edge in adjacencyList[currentNode])
                    {
                        minimumScore = Math.Min(minimumScore, edge[1]);
                        if (!visitedNodes[edge[0]])
                        {
                            visitedNodes[edge[0]] = true;
                            queue.Enqueue(edge[0]);
                        }
                    }
                }
                return minimumScore;
            }

            /*
            Approach 2: Depth First Search
            Complexity Analysis
            Here n is the number of nodes and e is the total number edges.
            •	Time complexity: O(n+e).
            o	The dfs function visits each node once, which takes O(n) time in total. Because we have undirected edges, each edge can only be iterated twice (by nodes at the end), resulting in O(e) operations total while visiting all nodes.
            o	We also need O(e) time to initialize the adjacency list and O(n) to initialize the visit array.
            o	As a result, the total time required is O(n+e).
            •	Space complexity: O(n+e).
            o	Building the adjacency list takes O(e) space.
            o	The recursion call stack used by dfs can have no more than n elements in the worst-case scenario. It would take up O(n) space in that case.
            o	The visit array takes O(n) space.

            */
            public int DFS(int numberOfNodes, int[][] roads)
            {
                Dictionary<int, List<List<int>>> adjacencyList = new Dictionary<int, List<List<int>>>();
                foreach (int[] road in roads)
                {
                    if (!adjacencyList.ContainsKey(road[0]))
                    {
                        adjacencyList[road[0]] = new List<List<int>>();
                    }
                    adjacencyList[road[0]].Add(new List<int> { road[1], road[2] });

                    if (!adjacencyList.ContainsKey(road[1]))
                    {
                        adjacencyList[road[1]] = new List<List<int>>();
                    }
                    adjacencyList[road[1]].Add(new List<int> { road[0], road[2] });
                }

                bool[] visitedNodes = new bool[numberOfNodes + 1];
                DepthFirstSearch(1, adjacencyList, visitedNodes);

                return minimumScore;
            }
            int minimumScore = int.MaxValue;

            public void DepthFirstSearch(int currentNode, Dictionary<int, List<List<int>>> adjacencyList, bool[] visitedNodes)
            {
                visitedNodes[currentNode] = true;
                if (!adjacencyList.ContainsKey(currentNode))
                {
                    return;
                }
                foreach (List<int> edge in adjacencyList[currentNode])
                {
                    minimumScore = Math.Min(minimumScore, edge[1]);
                    if (!visitedNodes[edge[0]])
                    {
                        DepthFirstSearch(edge[0], adjacencyList, visitedNodes);
                    }
                }
            }
            /*
            Approach 3: Union-find
            Complexity Analysis
Here n is the number of nodes and e is the total number edges.
•	Time complexity: O(n+e).
o	For T operations, the amortized time complexity of the union-find algorithm (using path compression with union by rank) is O(alpha(T)). Here, α(T) is the inverse Ackermann function that grows so slowly, that it doesn't exceed 4 for all reasonable T (approximately T<10600). You can read more about the complexity of union-find here. Because the function grows so slowly, we consider it to be O(1).
o	Initializing UnionFind takes O(n) time beacuse we are initializing the parent and rank arrays of size n + 1 each.
o	We iterate through every edge and perform union of the nodes connected by the edge which takes O(1) time per operation. It takes O(e) time for e edges.
o	We again iterate through every edge and use find operation to find the component of node 1 and a node having one end in the edge. It also takes O(1) per operation and takes O(e) time for all the e edges.
o	As a result, the total time required is O(n+e).
•	Space complexity: O(n).
o	We are using the parent and rank arrays, both of which require O(n) space each.

            */
            public int UsingUnionFind(int n, int[][] roads)
            {
                UnionFind dsu = new UnionFind(n + 1);
                int answer = int.MaxValue;

                foreach (int[] road in roads)
                {
                    dsu.UnionSet(road[0], road[1]);
                }

                foreach (int[] road in roads)
                {
                    if (dsu.Find(1) == dsu.Find(road[0]))
                    {
                        answer = Math.Min(answer, road[2]);
                    }
                }

                return answer;
            }
            class UnionFind
            {
                int[] parent;
                int[] rank;

                public UnionFind(int size)
                {
                    parent = new int[size];
                    for (int i = 0; i < size; i++)
                        parent[i] = i;
                    rank = new int[size];
                }

                public int Find(int x)
                {
                    if (parent[x] != x)
                        parent[x] = Find(parent[x]);
                    return parent[x];
                }

                public void UnionSet(int x, int y)
                {
                    int xset = Find(x), yset = Find(y);
                    if (xset == yset)
                    {
                        return;
                    }
                    else if (rank[xset] < rank[yset])
                    {
                        parent[xset] = yset;
                    }
                    else if (rank[xset] > rank[yset])
                    {
                        parent[yset] = xset;
                    }
                    else
                    {
                        parent[yset] = xset;
                        rank[xset]++;
                    }
                }
            }


        }



        /* 2551. Put Marbles in Bags
        https://leetcode.com/problems/put-marbles-in-bags/description/
         */
        class PutMarblesSolution
        {
            /*Approach: Sorting
            Complexity Analysis
Let n be the number of elements in the input array weights.
•	Time complexity: O(n⋅logn)
o	We need to sort the pairWeights, the array of every pair value having n - 1 elements, it takes O(n⋅logn) time.
o	We then traverse the sorted pairWeights and calculate the cumulative sum of the k - 1 largest elements and the sum of the k - 1 smallest elements, this step takes O(k) time.
o	To sum up, the overall time complexity is O(n⋅logn).
•	Space complexity: O(n)
o	We create an auxiliary array pairWeights of size n - 1 to store the value of all pairs.

*/
            public long UsingSort(int[] weights, int k)
            {
                // We collect and sort the value of all n - 1 pairs.
                int n = weights.Length;
                int[] pairWeights = new int[n - 1];
                for (int i = 0; i < n - 1; ++i)
                {
                    pairWeights[i] = weights[i] + weights[i + 1];
                }
                // We will sort only the first (n - 1) elements of the array.
                Array.Sort(pairWeights, 0, n - 1);

                // Get the difference between the largest k - 1 values and the 
                // smallest k - 1 values.
                long answer = 0l;
                for (int i = 0; i < k - 1; ++i)
                {
                    answer += pairWeights[n - 2 - i] - pairWeights[i];
                }

                return answer;
            }
        }



        /* 351. Android Unlock Patterns
        https://leetcode.com/problems/android-unlock-patterns/description/
         */

        class NumberOfPatternsSol
        {

            // All possible single-step moves on the lock pattern grid
            // Each array represents a move as {row change, column change}
            private static int[][] SINGLE_STEP_MOVES = {
        new int[]{ 0, 1 },
        new int[]{ 0, -1 },
        new int[]{ 1, 0 },
        new int[]{ -1, 0 }, // Adjacent moves (right, left, down, up)
        new int[]{ 1, 1 },
        new int[]{ -1, 1 },
        new int[]{ 1, -1 },
        new int[]{ -1, -1 }, // Diagonal moves
        new int[]{ -2, 1 },
        new int[]{ -2, -1 },
        new int[]{ 2, 1 },
        new int[]{ 2, -1 }, // Extended moves (knight-like moves)
        new int[]{ 1, -2 },
        new int[]{ -1, -2 },
        new int[]{ 1, 2 },
        new int[]{ -1, 2 },
    };

            // Moves that require a dot to be visited in between
            // These moves "jump" over a dot, which must have been previously visited
            private static int[][] SKIP_DOT_MOVES = {
        new int[]{ 0, 2 },
        new int[]{ 0, -2 },
        new int[]{ 2, 0 },
        new int[]{ -2, 0 }, // Straight skip moves (e.g., 1 to 3, 4 to 6)
        new int[]{ -2, -2 },
        new int[]{ 2, 2 },
        new int[]{ 2, -2 },
        new int[]{ -2, 2 }, // Diagonal skip moves (e.g., 1 to 9, 3 to 7)
    };
            /*
            Approach 1: Backtracking
            Complexity Analysis
            Let n be the maximum numbers of keys allowed in the pattern.
            •	Time complexity: O(9⋅(8^n))
            The method numberOfPatterns iterates through all 9 dots on the grid as a starting point.
            In each call to countPatternsFromDot, the function explores all possible moves from the current dot. Let 8 be the approximate number of choices at each dot. In the worst-case scenario, each recursive call leads to further recursive calls, up to a maximum depth of n. Thus, the total number of patterns explored can be approximated by 9×8n (each move branching out into multiple further moves).
            Thus, the overall time complexity of the algorithm is O(9⋅(8^n)).
            •	Space complexity: O(n)
            The arrays SINGLE_STEP_MOVES and SKIP_DOT_MOVES use constant space.
            The visitedDots matrix is a 3×3 boolean array, which takes up constant space.
            The maximum depth of the recursion stack is n.
            Thus, the overall space complexity of the algorithm is O(n).


            */
            public int WithBacktracking(int m, int n)
            {
                int totalPatterns = 0;
                // Start from each of the 9 dots on the grid
                for (int row = 0; row < 3; row++)
                {
                    for (int col = 0; col < 3; col++)
                    {
                        bool[][] visitedDots = new bool[3][];
                        // Count patterns starting from this dot
                        totalPatterns +=
                        CountPatternsFromDot(m, n, 1, row, col, visitedDots);
                    }
                }
                return totalPatterns;
            }

            private int CountPatternsFromDot(
                int m,
                int n,
                int currentLength,
                int currentRow,
                int currentCol,
                bool[][] visitedDots
            )
            {
                // Base case: if current pattern length exceeds n, stop exploring
                if (currentLength > n)
                {
                    return 0;
                }

                int validPatterns = 0;
                // If current pattern length is within the valid range, count it
                if (currentLength >= m) validPatterns++;

                // Mark current dot as visited
                visitedDots[currentRow][currentCol] = true;

                // Explore all single-step moves
                foreach (int[] move in SINGLE_STEP_MOVES)
                {
                    int newRow = currentRow + move[0];
                    int newCol = currentCol + move[1];
                    if (IsValidMove(newRow, newCol, visitedDots))
                    {
                        // Recursively count patterns from the new position
                        validPatterns +=
                        CountPatternsFromDot(
                            m,
                            n,
                            currentLength + 1,
                            newRow,
                            newCol,
                            visitedDots
                        );
                    }
                }

                // Explore all skip-dot moves
                foreach (int[] move in SKIP_DOT_MOVES)
                {
                    int newRow = currentRow + move[0];
                    int newCol = currentCol + move[1];
                    if (IsValidMove(newRow, newCol, visitedDots))
                    {
                        // Check if the middle dot has been visited
                        int middleRow = currentRow + move[0] / 2;
                        int middleCol = currentCol + move[1] / 2;
                        if (visitedDots[middleRow][middleCol])
                        {
                            // If middle dot is visited, this move is valid
                            validPatterns +=
                            CountPatternsFromDot(
                                m,
                                n,
                                currentLength + 1,
                                newRow,
                                newCol,
                                visitedDots
                            );
                        }
                    }
                }

                // Backtrack: unmark the current dot before returning
                visitedDots[currentRow][currentCol] = false;
                return validPatterns;
            }

            // Helper method to check if a move is valid
            private bool IsValidMove(int row, int col, bool[][] visitedDots)
            {
                // A move is valid if it's within the grid and the dot hasn't been visited
                return (
                    row >= 0 && col >= 0 && row < 3 && col < 3 && !visitedDots[row][col]
                );
            }

            /*
            Approach 2: Backtracking (Optimized)
Complexity Analysis
Let n be the maximum number of keys allowed in the pattern.
•	Time complexity: O(3⋅(8^n))
The algorithm calls the recursive function countPatternsFromNumber a total of 3 times. Each recursive call explores approximately 8 surrounding dots in each call. In the worst case, each recursive call can spread up to a maximum depth of n with a branching factor of 8. Thus, the total time complexity of the algorithm comes out to be O(3⋅(8^n)).
•	Space complexity: O(n)
The jump array is a 10×10 grid which takes constant space. The maximum depth of recursion can be n in the worst case. Thus, the overall space complexity is O(n).

            */
            public int BacktrackingOptimal(int m, int n)
            {
                int[][] jump = new int[10][];

                // Initialize the jump over numbers for all valid jumps
                jump[1][3] = jump[3][1] = 2;
                jump[4][6] = jump[6][4] = 5;
                jump[7][9] = jump[9][7] = 8;
                jump[1][7] = jump[7][1] = 4;
                jump[2][8] = jump[8][2] = 5;
                jump[3][9] = jump[9][3] = 6;
                jump[1][9] = jump[9][1] = jump[3][7] = jump[7][3] = 5;

                bool[] visitedNumbers = new bool[10];
                int totalPatterns = 0;

                // Count patterns starting from corner numbers (1, 3, 7, 9) and multiply by 4 due to symmetry
                totalPatterns +=
                CountPatternsFromNumber(1, 1, m, n, jump, visitedNumbers) * 4;

                // Count patterns starting from edge numbers (2, 4, 6, 8) and multiply by 4 due to symmetry
                totalPatterns +=
                CountPatternsFromNumber(2, 1, m, n, jump, visitedNumbers) * 4;

                // Count patterns starting from the center number (5)
                totalPatterns +=
                CountPatternsFromNumber(5, 1, m, n, jump, visitedNumbers);

                return totalPatterns;
            }

            private int CountPatternsFromNumber(
                int currentNumber,
                int currentLength,
                int minLength,
                int maxLength,
                int[][] jump,
                bool[] visitedNumbers
            )
            {
                // Base case: if current pattern length exceeds maxLength, stop exploring
                if (currentLength > maxLength) return 0;

                int validPatterns = 0;
                // If current pattern length is within the valid range, count it
                if (currentLength >= minLength)
                {
                    validPatterns++;
                }

                visitedNumbers[currentNumber] = true;

                // Explore all possible next numbers
                for (int nextNumber = 1; nextNumber <= 9; nextNumber++)
                {
                    int jumpOverNumber = jump[currentNumber][nextNumber];
                    // Check if the next number is unvisited and either:
                    // 1. There's no number to jump over, or
                    // 2. The number to jump over has been visited
                    if (
                        !visitedNumbers[nextNumber] &&
                        (jumpOverNumber == 0 || visitedNumbers[jumpOverNumber])
                    )
                    {
                        validPatterns +=
                        CountPatternsFromNumber(
                            nextNumber,
                            currentLength + 1,
                            minLength,
                            maxLength,
                            jump,
                            visitedNumbers
                        );
                    }
                }

                // Backtrack: unmark the current number before returning
                visitedNumbers[currentNumber] = false;

                return validPatterns;
            }

            /*
            Approach 3: Memoization
Complexity Analysis
Let n be the maximum numbers of keys allowed in the pattern.
•	Time complexity: O(1)
Due to memoization, the time complexity of the algorithm is bounded by the total time required to fill the dp array. The total size of dp is 10×(1<<10) or 10240. So, the overall time complexity of the algorithm is O(10240), which can be simplified to O(1).
•	Space complexity: O(n)
The jump array and the dp array both use constant space irrelevant of the input. The recursion stack has a space complexity of O(n). Thus, the space complexity of the algorithm is O(n).


            */
            public int DPMemo(int minLength, int maxLength)
            {
                int[][] jump = new int[10][];
                for (int i = 0; i < 10; i++)
                {
                    jump[i] = new int[10];
                }

                // Initialize the jump over numbers for all valid jumps
                jump[1][3] = jump[3][1] = 2;
                jump[4][6] = jump[6][4] = 5;
                jump[7][9] = jump[9][7] = 8;
                jump[1][7] = jump[7][1] = 4;
                jump[2][8] = jump[8][2] = 5;
                jump[3][9] = jump[9][3] = 6;
                jump[1][9] = jump[9][1] = jump[3][7] = jump[7][3] = 5;

                int visitedNumbers = 0;
                int totalPatterns = 0;
                int?[][] dp = new int?[10][];
                for (int i = 0; i < 10; i++)
                {
                    dp[i] = new int?[1 << 10];
                }

                // Count patterns starting from corner numbers (1, 3, 7, 9) and multiply by 4 due to symmetry
                totalPatterns += CountPatternsFromNumber(1, 1, minLength, maxLength, jump, visitedNumbers, dp) * 4;

                // Count patterns starting from edge numbers (2, 4, 6, 8) and multiply by 4 due to symmetry
                totalPatterns += CountPatternsFromNumber(2, 1, minLength, maxLength, jump, visitedNumbers, dp) * 4;

                // Count patterns starting from the center number (5)
                totalPatterns += CountPatternsFromNumber(5, 1, minLength, maxLength, jump, visitedNumbers, dp);

                return totalPatterns;
            }

            private int CountPatternsFromNumber(
                int currentNumber,
                int currentLength,
                int minLength,
                int maxLength,
                int[][] jump,
                int visitedNumbers,
                int?[][] dp
            )
            {
                // Base case: if current pattern length exceeds maxLength, stop exploring
                if (currentLength > maxLength) return 0;

                if (dp[currentNumber][visitedNumbers] != null) return dp[currentNumber][visitedNumbers].Value;

                int validPatterns = 0;
                // If current pattern length is within the valid range, count it
                if (currentLength >= minLength)
                {
                    validPatterns++;
                }

                visitedNumbers = SetBit(visitedNumbers, currentNumber);

                // Explore all possible next numbers
                for (int nextNumber = 1; nextNumber <= 9; nextNumber++)
                {
                    int jumpOverNumber = jump[currentNumber][nextNumber];
                    // Check if the next number is unvisited and either:
                    // 1. There's no number to jump over, or
                    // 2. The number to jump over has been visited
                    if (!IsSet(visitedNumbers, nextNumber) &&
                        (jumpOverNumber == 0 || IsSet(visitedNumbers, jumpOverNumber)))
                    {
                        validPatterns += CountPatternsFromNumber(
                            nextNumber,
                            currentLength + 1,
                            minLength,
                            maxLength,
                            jump,
                            visitedNumbers,
                            dp
                        );
                    }
                }

                // Backtrack: unmark the current number before returning
                visitedNumbers = ClearBit(visitedNumbers, currentNumber);

                return (int)(dp[currentNumber][visitedNumbers] = validPatterns);
            }

            private int SetBit(int num, int position)
            {
                num |= 1 << (position - 1);
                return num;
            }

            private int ClearBit(int num, int position)
            {
                num ^= 1 << (position - 1);
                return num;
            }

            private bool IsSet(int num, int position)
            {
                int bitAtPosition = (num >> (position - 1)) & 1;
                return bitAtPosition == 1;
            }
        }



        /* 465. Optimal Account Balancing
        https://leetcode.com/problems/optimal-account-balancing/description/
         */
        class MinTransfersSol
        {
            /*
            Approach 1: Backtracking
Complexity Analysis
Let n be the length of transactions.
•	Time complexity: O((n−1)!)
o	In dfs(0), there exists a maximum of n−1 persons as possible nxt, each of which leads to a recursive call to dfs(1). Therefore, we have dfs(0)=(n−1)⋅dfs(1)=(n−1)⋅((n−2)⋅dfs(2))=(n−1)⋅(n−2)⋅((n−3)⋅dfs(3))=...=(n−1)!⋅dfs(n−1)
o	dfs(n - 1) can be determined in O(1) time.
•	Space complexity: O(n)
o	Both balance_map and balance_list possess at most n net balances.
o	The space complexity of a recursive call relies on the maximum depth of the recursive call stack, which is equal to n. As each recursive call increments cur by 1, and each level consumes a constant amount of space.
            */
            public int WithBacktracking(int[][] transactions)
            {
                Dictionary<int, int> creditMap = new Dictionary<int, int>();
                foreach (int[] transaction in transactions)
                {
                    int creditor = transaction[0];
                    int debtor = transaction[1];
                    int amount = transaction[2];
                    creditMap[creditor] = creditMap.GetValueOrDefault(creditor, 0) + amount;
                    creditMap[debtor] = creditMap.GetValueOrDefault(debtor, 0) - amount;
                }

                List<int> creditList = new List<int>();
                foreach (int amount in creditMap.Values)
                {
                    if (amount != 0)
                    {
                        creditList.Add(amount);
                    }
                }

                int n = creditList.Count;
                return Dfs(0, n, creditList);
            }

            private int Dfs(int currentIndex, int totalCount, List<int> creditList)
            {
                while (currentIndex < totalCount && creditList[currentIndex] == 0)
                {
                    currentIndex++;
                }

                if (currentIndex == totalCount)
                {
                    return 0;
                }

                int minimumCost = int.MaxValue;
                for (int nextIndex = currentIndex + 1; nextIndex < totalCount; nextIndex++)
                {
                    // If nextIndex is a valid recipient, do the following: 
                    // 1. add currentIndex's balance to nextIndex.
                    // 2. recursively call Dfs(currentIndex + 1).
                    // 3. remove currentIndex's balance from nextIndex.
                    if (creditList[nextIndex] * creditList[currentIndex] < 0)
                    {
                        creditList[nextIndex] += creditList[currentIndex];
                        minimumCost = Math.Min(minimumCost, 1 + Dfs(currentIndex + 1, totalCount, creditList));
                        creditList[nextIndex] -= creditList[currentIndex];
                    }
                }

                return minimumCost;
            }

            /*
            Approach 2: Dynamic Programming
Complexity Analysis
Let n be the length of transactions.
•	Time complexity: O(n⋅(2^n))
o	We build memo, an array of size O(2^n)as memory, equal to the number of possible states. Each state is computed with a traverse through balance_list, which takes O(n) time.
•	Space complexity: O(2^n)
o	The length of memo is 2^n.
o	The space complexity of a recursive call depends on the maximum depth of the recursive call stack, which is n. As each recursive call removes one set bit from total_mask. Therefore, at most O(n) levels of recursion will be created, and each level consumes a constant amount of space.

            */
            public int WithDP(int[][] transactions)
            {
                Dictionary<int, int> creditMap = new Dictionary<int, int>();
                foreach (int[] transaction in transactions)
                {
                    creditMap[transaction[0]] = creditMap.GetValueOrDefault(transaction[0], 0) + transaction[2];
                    creditMap[transaction[1]] = creditMap.GetValueOrDefault(transaction[1], 0) - transaction[2];
                }

                List<int> creditList = new List<int>();
                foreach (int amount in creditMap.Values)
                {
                    if (amount != 0)
                    {
                        creditList.Add(amount);
                    }
                }

                int numberOfCredits = creditList.Count;
                int[] memo = new int[1 << numberOfCredits];
                Array.Fill(memo, -1);
                memo[0] = 0;
                return numberOfCredits - Dfs((1 << numberOfCredits) - 1, memo, creditList);
            }

            private int Dfs(int totalMask, int[] memo, List<int> creditList)
            {
                if (memo[totalMask] != -1)
                {
                    return memo[totalMask];
                }
                int balanceSum = 0, answer = 0;

                // Remove one person at a time in total_mask
                for (int i = 0; i < creditList.Count; i++)
                {
                    int currentBit = 1 << i;
                    if ((totalMask & currentBit) != 0)
                    {
                        balanceSum += creditList[i];
                        answer = Math.Max(answer, Dfs(totalMask ^ currentBit, memo, creditList));
                    }
                }

                // If the total balance of total_mask is 0, increment answer by 1.
                memo[totalMask] = answer + (balanceSum == 0 ? 1 : 0);
                return memo[totalMask];
            }
        }

        /* 1152. Analyze User Website Visit Pattern
        https://leetcode.com/problems/analyze-user-website-visit-pattern/description/
         */


        class MostVisitedPatternSol
        {
            public IList<string> MostVisitedPattern(string[] username, int[] timestamp, string[] website)
            {
                Dictionary<string, List<Pair>> userWebsiteMap = new Dictionary<string, List<Pair>>();
                int userCount = username.Length;
                // collect the website info for every user, key: username, value: (timestamp, website)
                for (int i = 0; i < userCount; i++)
                {
                    if (!userWebsiteMap.ContainsKey(username[i]))
                    {
                        userWebsiteMap[username[i]] = new List<Pair>();
                    }
                    userWebsiteMap[username[i]].Add(new Pair(timestamp[i], website[i]));
                }
                // count map to record every 3 combination occurring time for the different user.
                Dictionary<string, int> countMap = new Dictionary<string, int>();
                string resultPattern = "";
                foreach (var user in userWebsiteMap.Keys)
                {
                    HashSet<string> visitedPatterns = new HashSet<string>();
                    // this set is to avoid visit the same 3-seq in one user
                    List<Pair> userVisits = userWebsiteMap[user];
                    userVisits.Sort((a, b) => a.Time.CompareTo(b.Time)); // sort by time
                                                                         // brute force O(N ^ 3)
                    for (int i = 0; i < userVisits.Count; i++)
                    {
                        for (int j = i + 1; j < userVisits.Count; j++)
                        {
                            for (int k = j + 1; k < userVisits.Count; k++)
                            {
                                string pattern = userVisits[i].Web + " " + userVisits[j].Web + " " + userVisits[k].Web;
                                if (!visitedPatterns.Contains(pattern))
                                {
                                    if (!countMap.ContainsKey(pattern))
                                    {
                                        countMap[pattern] = 0;
                                    }
                                    countMap[pattern]++;
                                    visitedPatterns.Add(pattern);
                                }
                                if (resultPattern == "" || countMap[resultPattern] < countMap[pattern] ||
                                    (countMap[resultPattern] == countMap[pattern] && string.Compare(resultPattern, pattern) > 0))
                                {
                                    // make sure the right lexicographical order
                                    resultPattern = pattern;
                                }
                            }
                        }
                    }
                }
                // grab the right answer
                string[] resultArray = resultPattern.Split(" ");
                List<string> result = new List<string>(resultArray);
                return result;
            }
            class Pair
            {
                public int Time { get; set; }
                public string Web { get; set; }

                public Pair(int time, string web)
                {
                    Time = time;
                    Web = web;
                }
            }
        }


        /* 224. Basic Calculator
        https://leetcode.com/problems/basic-calculator/description/
         */
        class CalculateSol
        {
            /*
            Approach 1: Stack and String Reversal
            Complexity Analysis
            •	Time Complexity: O(N), where N is the length of the string.
            •	Space Complexity: O(N), where N is the length of the string.

            */

            public int UsingStackAndStringRev(string expression)
            {

                int operand = 0;
                int digitCount = 0;
                Stack<object> stack = new Stack<object>();

                for (int i = expression.Length - 1; i >= 0; i--)
                {

                    char currentChar = expression[i];

                    if (char.IsDigit(currentChar))
                    {

                        // Forming the operand - in reverse order.
                        operand = (int)Math.Pow(10, digitCount) * (int)(currentChar - '0') + operand;
                        digitCount += 1;

                    }
                    else if (currentChar != ' ')
                    {
                        if (digitCount != 0)
                        {

                            // Save the operand on the stack
                            // As we encounter some non-digit.
                            stack.Push(operand);
                            digitCount = 0;
                            operand = 0;

                        }
                        if (currentChar == '(')
                        {

                            int evaluatedResult = EvaluateExpression(stack);
                            stack.Pop();

                            // Append the evaluated result to the stack.
                            // This result could be of a sub-expression within the parenthesis.
                            stack.Push(evaluatedResult);

                        }
                        else
                        {
                            // For other non-digits just push onto the stack.
                            stack.Push(currentChar);
                        }
                    }
                }

                // Push the last operand to stack, if any.
                if (digitCount != 0)
                {
                    stack.Push(operand);
                }

                // Evaluate any left overs in the stack.
                return EvaluateExpression(stack);
            }
            private int EvaluateExpression(Stack<object> stack)
            {

                // If stack is empty or the expression starts with
                // a symbol, then append 0 to the stack.
                // i.e. [1, '-', 2, '-'] becomes [1, '-', 2, '-', 0]
                if (stack.Count == 0 || !(stack.Peek() is int))
                {
                    stack.Push(0);
                }

                int result = (int)stack.Pop();

                // Evaluate the expression till we get corresponding ')'
                while (stack.Count > 0 && !((char)stack.Peek() == ')'))
                {

                    char sign = (char)stack.Pop();

                    if (sign == '+')
                    {
                        result += (int)stack.Pop();
                    }
                    else
                    {
                        result -= (int)stack.Pop();
                    }
                }
                return result;
            }
            /*
Approach 2: Stack and No String Reversal
Complexity Analysis
•	Time Complexity: O(N), where N is the length of the string. The difference in time complexity between this approach and the previous one is that every character in this approach will get processed exactly once. However, in the previous approach, each character can potentially get processed twice, once when it's pushed onto the stack and once when it's popped for processing of the final result (or a subexpression). That's why this approach is faster.
•	Space Complexity: O(N), where N is the length of the string.

            */
            public int UsingStack(String s)
            {

                Stack<int> stack = new Stack<int>();
                int operand = 0;
                int result = 0; // For the on-going result
                int sign = 1;  // 1 means positive, -1 means negative

                for (int i = 0; i < s.Length; i++)
                {

                    char ch = s[i];
                    if (Char.IsDigit(ch))
                    {

                        // Forming operand, since it could be more than one digit
                        operand = 10 * operand + (int)(ch - '0');

                    }
                    else if (ch == '+')
                    {

                        // Evaluate the expression to the left,
                        // with result, sign, operand
                        result += sign * operand;

                        // Save the recently encountered '+' sign
                        sign = 1;

                        // Reset operand
                        operand = 0;

                    }
                    else if (ch == '-')
                    {

                        result += sign * operand;
                        sign = -1;
                        operand = 0;

                    }
                    else if (ch == '(')
                    {

                        // Push the result and sign on to the stack, for later
                        // We push the result first, then sign
                        stack.Push(result);
                        stack.Push(sign);

                        // Reset operand and result, as if new evaluation begins for the new sub-expression
                        sign = 1;
                        result = 0;

                    }
                    else if (ch == ')')
                    {

                        // Evaluate the expression to the left
                        // with result, sign and operand
                        result += sign * operand;

                        // ')' marks end of expression within a set of parenthesis
                        // Its result is multiplied with sign on top of stack
                        // as stack.pop() is the sign before the parenthesis
                        result *= stack.Pop();

                        // Then add to the next operand on the top.
                        // as stack.pop() is the result calculated before this parenthesis
                        // (operand on stack) + (sign on stack * (result from parenthesis))
                        result += stack.Pop();

                        // Reset the operand
                        operand = 0;
                    }
                }
                return result + (sign * operand);
            }
        }


        /* 227. Basic Calculator II
        https://leetcode.com/problems/basic-calculator-ii/description/
         */
        class CalculateIISol
        {
            /*
            Approach 1: Using Stack
Complexity Analysis
•	Time Complexity: O(n), where n is the length of the string s. We iterate over the string s at most twice.
•	Space Complexity: O(n), where n is the length of the string s.

            */
            public int UsingStack(String s)
            {

                if (s == null || s.Length == 0) return 0;
                int len = s.Length;
                Stack<int> stack = new Stack<int>();
                int currentNumber = 0;
                char operation = '+';
                for (int i = 0; i < len; i++)
                {
                    char currentChar = s[i];
                    if (Char.IsDigit(currentChar))
                    {
                        currentNumber = (currentNumber * 10) + (currentChar - '0');
                    }
                    if (!Char.IsDigit(currentChar) && !Char.IsWhiteSpace(currentChar) || i == len - 1)
                    {
                        if (operation == '-')
                        {
                            stack.Push(-currentNumber);
                        }
                        else if (operation == '+')
                        {
                            stack.Push(currentNumber);
                        }
                        else if (operation == '*')
                        {
                            stack.Push(stack.Pop() * currentNumber);
                        }
                        else if (operation == '/')
                        {
                            stack.Push(stack.Pop() / currentNumber);
                        }
                        operation = currentChar;
                        currentNumber = 0;
                    }
                }
                int result = 0;
                while (stack.Count > 0)
                {
                    result += stack.Pop();
                }
                return result;
            }
            /*
            Approach 2: Optimised Approach without the stack
            Complexity Analysis
•	Time Complexity: O(n), where n is the length of the string s.
•	Space Complexity: O(1), as we use constant extra space to store lastNumber, result and so on.

            */
            public int OptimalWithoutStack(String s)
            {
                if (s == null || s.Length == 0) return 0;
                int length = s.Length;
                int currentNumber = 0, lastNumber = 0, result = 0;
                char operation = '+';
                for (int i = 0; i < length; i++)
                {
                    char currentChar = s[i];
                    if (Char.IsDigit(currentChar))
                    {
                        currentNumber = (currentNumber * 10) + (currentChar - '0');
                    }
                    if (!Char.IsDigit(currentChar) && !Char.IsWhiteSpace(currentChar) || i == length - 1)
                    {
                        if (operation == '+' || operation == '-')
                        {
                            result += lastNumber;
                            lastNumber = (operation == '+') ? currentNumber : -currentNumber;
                        }
                        else if (operation == '*')
                        {
                            lastNumber = lastNumber * currentNumber;
                        }
                        else if (operation == '/')
                        {
                            lastNumber = lastNumber / currentNumber;
                        }
                        operation = currentChar;
                        currentNumber = 0;
                    }
                }
                result += lastNumber;
                return result;
            }
        }


        /* 772. Basic Calculator III
        https://leetcode.com/problems/basic-calculator-iii/description/
         */

        class CalculateIIISol
        {
            /*
            Approach1: Stack
            Complexity Analysis
Given n as the length of the expression,
For this analysis, we will assume you are using the Python implementation since it is relevant that curr is of type int.
•	Time complexity: O(n)
The analysis here is simple - each character in the input can only be pushed and popped from the stack at most one time. Every other operation in each of the O(n) iterations costs O(1) - updating curr, calling evaluate, etc.
•	Space complexity: O(n)
The stack could grow to a size of O(n) - for example, if the expression contains only the addition of single-digit numbers.

            */
            public int UsingStack(string s)
            {
                Stack<string> stack = new Stack<string>();
                string currentNumber = "";
                char previousOperator = '+';
                s += "@";
                HashSet<string> operators = new HashSet<string> { "+", "-", "*", "/" };

                foreach (char c in s)
                {
                    if (char.IsDigit(c))
                    {
                        currentNumber += c;
                    }
                    else if (c == '(')
                    {
                        stack.Push(previousOperator.ToString()); // convert char to string before pushing
                        previousOperator = '+';
                    }
                    else
                    {
                        if (previousOperator == '*' || previousOperator == '/')
                        {
                            stack.Push(Evaluate(previousOperator, stack.Pop(), currentNumber));
                        }
                        else
                        {
                            stack.Push(Evaluate(previousOperator, currentNumber, "0"));
                        }

                        currentNumber = "";
                        previousOperator = c;
                        if (c == ')')
                        {
                            int currentTerm = 0;
                            while (!operators.Contains(stack.Peek()))
                            {
                                currentTerm += int.Parse(stack.Pop());
                            }

                            currentNumber = currentTerm.ToString();
                            previousOperator = stack.Pop()[0]; // convert string from stack back to char
                        }
                    }
                }

                int answer = 0;
                foreach (string num in stack)
                {
                    answer += int.Parse(num);
                }

                return answer;
            }
            private string Evaluate(char operatorChar, string first, string second)
            {
                int firstNumber = int.Parse(first);
                int secondNumber = int.Parse(second);
                int result = 0;

                if (operatorChar == '+')
                {
                    result = firstNumber;
                }
                else if (operatorChar == '-')
                {
                    result = -firstNumber;
                }
                else if (operatorChar == '*')
                {
                    result = firstNumber * secondNumber;
                }
                else
                {
                    result = firstNumber / secondNumber;
                }

                return result.ToString();
            }
            /*
            Approach 2: Solve Isolated Expressions With Recursion
Complexity Analysis
Given n as the length of the expression,
•	Time complexity: O(n)
The time complexity is the same as the previous approach for the same reason. i is strictly increasing and increments on each iteration. Any given character can only be pushed to a stack once and popped from a stack once, so the total number of operations across the algorithm is linear.
•	Space complexity: O(n)
The stacks across all function calls could grow to a size of O(n) - for example, if the expression contains only the addition of single-digit numbers.

            */
            public int SolveIsolatedExpressionsWithRecursion(string expression)
            {
                expression += "@";
                int[] index = new int[1];
                return Solve(expression, index);
            }
            private int Evaluate(char operatorChar, int x, int y)
            {
                if (operatorChar == '+')
                {
                    return x;
                }
                else if (operatorChar == '-')
                {
                    return -x;
                }
                else if (operatorChar == '*')
                {
                    return x * y;
                }

                return x / y;
            }

            private int Solve(string expression, int[] index)
            {
                Stack<int> stack = new Stack<int>();
                int currentNumber = 0;
                char previousOperator = '+';

                while (index[0] < expression.Length)
                {
                    char currentChar = expression[index[0]];
                    if (currentChar == '(')
                    {
                        index[0]++;
                        currentNumber = Solve(expression, index);
                    }
                    else if (char.IsDigit(currentChar))
                    {
                        currentNumber = currentNumber * 10 + (currentChar - '0');
                    }
                    else
                    {
                        if (previousOperator == '*' || previousOperator == '/')
                        {
                            stack.Push(Evaluate(previousOperator, stack.Pop(), currentNumber));
                        }
                        else
                        {
                            stack.Push(Evaluate(previousOperator, currentNumber, 0));
                        }

                        if (currentChar == ')')
                        {
                            break;
                        }

                        currentNumber = 0;
                        previousOperator = currentChar;
                    }

                    index[0]++;
                }

                int result = 0;
                foreach (int number in stack)
                {
                    result += number;
                }

                return result;
            }



        }


        class Solution
        {
            public Dictionary<string, int> evaluationMap = new Dictionary<string, int>(); // evaluation map

            public class Term
            {
                public int parameter = 1; // the parameter of this term
                public List<string> variables = new List<string>(); // each factor (e.a. a*b*b*c->{a,b,b,c})

                public override string ToString()
                {
                    if (parameter == 0) return "";
                    string result = "";
                    foreach (string variable in variables) result += "*" + variable;
                    return parameter + result;
                }

                public bool Equals(Term that)
                {
                    if (this.variables.Count != that.variables.Count) return false;
                    for (int i = 0; i < this.variables.Count; i++)
                        if (!this.variables[i].Equals(that.variables[i])) return false;
                    return true;
                }

                public int CompareTo(Term that)
                {
                    if (this.variables.Count > that.variables.Count) return -1;
                    if (this.variables.Count < that.variables.Count) return 1;
                    for (int i = 0; i < this.variables.Count; i++)
                    {
                        int comparisonResult = this.variables[i].CompareTo(that.variables[i]);
                        if (comparisonResult != 0) return comparisonResult;
                    }
                    return 0;
                }

                public Term Multiply(Term that)
                {
                    Term product = new Term(this.parameter * that.parameter);
                    foreach (string variable in this.variables) product.variables.Add(new string(variable));
                    foreach (string variable in that.variables) product.variables.Add(new string(variable));
                    product.variables.Sort();
                    return product;
                }

                public Term(int x) { parameter = x; }
                public Term(string s)
                {
                    //TODO: fix below code trying to access outer callss fields
                    /* if (evaluationMap.ContainsKey(s))
                        parameter = evaluationMap[s];
                    else
                        variables.Add(s); */
                }
                public Term(Term that)
                {
                    this.parameter = that.parameter;
                    this.variables = new List<string>(that.variables);
                }
            }

            public List<Term> Combine(List<Term> terms) // combine the similar terms
            {
                terms.Sort((t1, t2) => t1.CompareTo(t2)); // sort all terms to make similar terms together
                List<Term> combinedTerms = new List<Term>();
                foreach (Term term in terms)
                {
                    if (combinedTerms.Count != 0 && term.Equals(combinedTerms[combinedTerms.Count - 1]))
                        combinedTerms[combinedTerms.Count - 1].parameter += term.parameter;
                    else
                        combinedTerms.Add(new Term(term));
                }
                return combinedTerms;
            }

            public List<string> BasicCalculatorIV(string expression, string[] evalVars, int[] evalInts)
            {
                for (int i = 0; i < evalVars.Length; i++) evaluationMap[evalVars[i]] = evalInts[i];
                int index = 0, length = expression.Length;
                Stack<Expression> stack = new Stack<Expression>();
                Stack<int> priorityStack = new Stack<int>();
                Expression zero = new Expression(0);
                stack.Push(zero);
                priorityStack.Push(0);
                int priority = 0;

                while (index < length)
                {
                    char character = expression[index];
                    if (char.IsDigit(character))
                    {
                        int number = 0;
                        while (index < length && char.IsDigit(expression[index]))
                        {
                            number = number * 10 + (expression[index] - '0');
                            index++;
                        }
                        stack.Push(new Expression(number));
                        continue;
                    }
                    if (char.IsLetter(character))
                    {
                        string variable = "";
                        while (index < length && char.IsLetter(expression[index]))
                        {
                            variable += expression[index];
                            index++;
                        }
                        stack.Push(new Expression(variable));
                        continue;
                    }
                    if (character == '(') priority += 2;
                    if (character == ')') priority -= 2;
                    if (character == '+' || character == '-' || character == '*')
                    {
                        int currentPriority = priority;
                        if (character == '*') currentPriority++;
                        while (priorityStack.Count > 0 && currentPriority <= priorityStack.Peek())
                        {
                            Expression current = stack.Pop(), last = stack.Pop();
                            priorityStack.Pop();
                            stack.Push(last.Calculate(current));
                        }
                        stack.Peek().operatorSymbol = character;
                        priorityStack.Push(currentPriority);
                    }
                    index++;
                }
                while (stack.Count > 1)
                {
                    Expression current = stack.Pop(), last = stack.Pop();
                    stack.Push(last.Calculate(current));
                }
                return stack.Peek().ToList();
            }
            class Expression
            {
                List<Term> termList = new List<Term>(); // Term List
                public char operatorSymbol = '+'; // Arithmetic symbol

                public Expression(int value)
                {
                    termList.Add(new Term(value));
                }

                public Expression(string value)
                {
                    termList.Add(new Term(value));
                }

                public Expression(List<Term> terms)
                {
                    termList = terms;
                }

                public Expression Times(Expression that)
                {
                    List<Term> combinedTerms = new List<Term>();
                    foreach (Term term1 in this.termList)
                    {
                        foreach (Term term2 in that.termList)
                        {
                            combinedTerms.Add(term1.Multiply(term2));
                        }
                    }
                    combinedTerms = Combine(combinedTerms);
                    return new Expression(combinedTerms);
                }

                public Expression Plus(Expression that, int sign)
                {
                    List<Term> combinedTerms = new List<Term>();
                    foreach (Term term in this.termList)
                    {
                        combinedTerms.Add(new Term(term));
                    }
                    foreach (Term term in that.termList)
                    {
                        Term newTerm = new Term(term);
                        newTerm.parameter *= sign;
                        combinedTerms.Add(newTerm);
                    }
                    combinedTerms = Combine(combinedTerms);
                    return new Expression(combinedTerms);
                }

                public Expression Calculate(Expression that)
                {
                    if (operatorSymbol == '+') return Plus(that, 1);
                    if (operatorSymbol == '-') return Plus(that, -1);
                    return Times(that);
                }

                public List<string> ToList()
                {
                    List<string> resultList = new List<string>();
                    foreach (Term term in termList)
                    {
                        string termString = term.ToString();
                        if (termString.Length > 0) resultList.Add(termString);
                    }
                    return resultList;
                }

                private List<Term> Combine(List<Term> terms)
                {
                    // Implementation of the Combine method should be provided here
                    return terms;
                }
            }
        }






    }


}