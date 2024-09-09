using System.Collections;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay
{
    public class StackProbs
    {
        //https://www.algoexpert.io/questions/balanced-brackets
        public static bool BalancedBrackets(string str)
        {
            //T:O(n)|S:O(n)
            string openingBrackets = "([{";
            string closingBrackets = ")]}";
            Dictionary<char, char> matchingBrackets = new Dictionary<char, char>();
            matchingBrackets.Add('(', ')');
            matchingBrackets.Add('[', ']');
            matchingBrackets.Add('{', '}');
            List<char> stack = new List<char>();
            for (int i = 0; i < str.Length; i++)
            {
                char letter = str[i];
                if (openingBrackets.IndexOf(letter) != -1)
                {
                    stack.Add(letter);
                }
                else if (closingBrackets.IndexOf(letter) != -1)
                {
                    if (stack.Count == 0) return false;

                    if (stack[stack.Count - 1] == matchingBrackets[letter])
                    {
                        stack.RemoveAt(stack.Count - 1);
                    }
                    else
                    {
                        return false;
                    }
                }
            }
            //Using Stack - 
            /*
            Stack<char> stack = new Stack<char>();
            foreach(char c in str){

                if(openBrak.IndexOf(c) !=-1)
                        stack.Push(c);
                else if(closingBrak.IndexOf(c) != -1)
                {
                    if(stack.Count == 0) return false;

                    if(matchBrak[stack.Peek()] != c) return false;
                    stack.Pop();                      
                }          
            }
            */
            return stack.Count == 0;
        }


        /*
        735. Asteroid Collision
https://leetcode.com/problems/asteroid-collision/

                https://www.algoexpert.io/questions/colliding-asteroids
        */

        public class CollidingAsteroidsSol
        {
            /*
            Approach: Stack
            Complexity Analysis
    Here, N is the number of asteroids in the list.
    •	Time complexity: O(N).
    We iterate over each asteroid in the list, and for each asteroid, we might iterate over the asteroids we have in the stack and keep popping until they explode. The important point is that each asteroid can be added and removed from the stack only once. Therefore, each asteroid can be processed only twice, first when we iterate over it and then again while popping it from the stack. Therefore, the total time complexity is equal to O(N).
    •	Space complexity: O(N).
    The only space required is for the stack; the maximum number of asteroids that could be there in the stack is N when there is no asteroid collision. The final list that we return, remainingAsteroids, is used to store the output, which is generally not considered part of space complexity. Hence, the total space complexity equals O(N).

            */
            public static int[] Stack(int[] asteroids)
            {
                //T:O(n) | S:O(n)
                if (asteroids.Length == 0) return new int[] { };

                Stack<int> asters = new Stack<int>();

                foreach (int aster in asteroids)
                {
                    if (asters.Count == 0 || aster > 0 || asters.Peek() < 0)
                    {
                        asters.Push(aster);
                        continue;
                    }
                    while (asters.Count > 0)
                    {
                        if (asters.Peek() < 0)
                        {
                            asters.Push(aster);
                            break;
                        }
                        //-3,5,-8
                        int prevAster = Math.Abs(asters.Peek());

                        // If the top asteroid in the stack is smaller, then it will explode.
                        // Hence pop it from the stack, also continue with the next asteroid in the stack.
                        if (prevAster > Math.Abs(aster)) { break; }
                        // If both asteroids have the same size, then both asteroids will explode.
                        // Pop the asteroid from the stack; also, we won't push the current asteroid to the stack.
                        if (prevAster == Math.Abs(aster)) { asters.Pop(); break; }

                        asters.Pop();

                        if (asters.Count == 0)
                        {
                            asters.Push(aster);
                            break;
                        }

                    }

                }
                int[] res = new int[asters.Count()];
                for (int i = asters.Count - 1; i >= 0; i--)
                {
                    res[i] = asters.Pop();
                }
                return res;
            }
        }

        /*2126. Destroying Asteroids	
        https://leetcode.com/problems/destroying-asteroids/description/
        */
        public class CanDestroyeAllAsteroidsSol
        {
            /*
            Approach: Greedy with Sorting
Time and Space Complexity
The time complexity of the code is O(n log n) where n is the number of asteroids. This is because the most time-consuming operation is the sort function, which typically has O(n log n) complexity.
The space complexity of the code is O(1) assuming that the sort is done in-place (as is typical with Python's sort on lists). This means that aside from a constant amount of additional space, the code does not require extra space that scales with the input size.

            */
            // Method to determine if all asteroids can be destroyed.
            public bool GreedyWithSort(int mass, int[] asteroids)
            {
                // Sort the asteroids array to process them in ascending order.
                Array.Sort(asteroids);

                // Use long to avoid integer overflow as mass might become larger than int range.
                long currentMass = mass;

                // Iterate through the sorted asteroids.
                foreach (int asteroidMass in asteroids)
                {
                    // If the current asteroid mass is larger than the current mass, destruction is not possible.
                    if (currentMass < asteroidMass)
                    {
                        return false;
                    }
                    // Otherwise, add the asteroid's mass to the current mass.
                    currentMass += asteroidMass;
                }

                // Return true if all asteroids have been successfully destroyed.
                return true;
            }

            //https://www.algoexpert.io/questions/longest-balanced-substring
            public int LongestBalancedSubstring(string str)
            {
                int maxLen = 0;

                //1. Naive/Bruteforce - Pair of loops and Stack
                //T:O(n^3) | S:O(n)
                maxLen = LongestBalancedSubstringNaive(str);

                //2. Optimal with Stack space
                //T:O(n) | S:O(n)
                maxLen = LongestBalancedSubstringOptimal1(str);

                //3. Optimal with NO auxiliary space
                //T:O(n) | S:O(1)
                maxLen = LongestBalancedSubstringOptimal2(str);

                //4. Optimal with NO auxiliary space, simplified version of #3
                //T:O(n) | S:O(1)
                maxLen = LongestBalancedSubstringOptimal3(str);

                return maxLen;
            }

            private int LongestBalancedSubstringOptimal3(string str)
            {
                return Math.Max(GetLongestBalancedDirection(str, true), GetLongestBalancedDirection(str, false));
            }

            private int GetLongestBalancedDirection(string str, bool leftToRight)
            {
                char openingParens = leftToRight ? '(' : ')';
                int strIdx = leftToRight ? 0 : str.Length - 1;
                int step = leftToRight ? 1 : -1;

                int maxLen = 0;
                int openingCount = 0, closingCount = 0;

                int idx = strIdx;
                while (idx >= 0 && idx < str.Length)
                {
                    char c = str[idx];

                    if (c == openingParens) openingCount++;
                    else closingCount++;

                    if (openingCount == closingCount)
                        maxLen = Math.Max(maxLen, closingCount * 2);
                    else if (closingCount > openingCount)
                    {
                        openingCount = 0;
                        closingCount = 0;
                    }
                    idx += step;
                }
                return maxLen;
            }

            private int LongestBalancedSubstringOptimal2(string str)
            {
                int maxLen = 0;
                int openingCount = 0, closingCount = 0;

                for (int i = 0; i < str.Length; i++)
                {
                    char c = str[i];

                    if (c == '(')
                    {
                        openingCount += 1;
                    }
                    else closingCount += 1;

                    if (openingCount == closingCount)
                        maxLen = Math.Max(maxLen, closingCount * 2);
                    else if (closingCount > openingCount)
                    {
                        openingCount = 0;
                        closingCount = 0;
                    }
                }
                openingCount = 0;
                closingCount = 0;
                //scenario: ((())( where opening brackets are more than closing one and still a valid substring exists
                for (int i = str.Length - 1; i >= 0; i--)
                {
                    char c = str[i];

                    if (c == '(')
                    {
                        openingCount++;
                    }
                    else closingCount++;

                    if (openingCount == closingCount)
                        maxLen = Math.Max(maxLen, openingCount * 2);
                    else if (openingCount > closingCount)
                    {
                        openingCount = 0;
                        closingCount = 0;
                    }
                }
                return maxLen;
            }

            private int LongestBalancedSubstringOptimal1(string str)
            {
                int maxLen = 0;
                Stack<int> idxStack = new Stack<int>();
                idxStack.Push(-1);

                for (int i = 0; i < str.Length; i++)
                {
                    if (str[i] == '(')
                    {
                        idxStack.Push(i);
                    }
                    else
                    {
                        idxStack.Pop(); //-1 is there by-default
                        if (idxStack.Count == 0)
                            idxStack.Push(i);
                        else
                        {
                            int balancedSubstringStartIdx = idxStack.Peek();
                            int currentLen = i - balancedSubstringStartIdx;
                            maxLen = Math.Max(maxLen, currentLen);
                        }
                    }
                }
                return maxLen;
            }

            private int LongestBalancedSubstringNaive(string str)
            {
                int maxLen = 0;
                for (int i = 0; i < str.Length; i++)
                {
                    for (int j = i + 2; j < str.Length + 1; j++)
                    {
                        if (IsBalanced(str.Substring(i, j - i)))
                        {
                            int currentLen = j - i;
                            maxLen = Math.Max(currentLen, maxLen);
                        }
                    }
                }
                return maxLen;
            }

            private bool IsBalanced(string str)
            {
                Stack<char> openParamsStack = new Stack<char>();

                for (int i = 0; i < str.Length; i++)
                {
                    char c = str[i];
                    if (c == '(')
                    {
                        openParamsStack.Push('(');
                    }
                    else if (openParamsStack.Count > 0)
                    {
                        openParamsStack.Pop();
                    }
                    else
                    {
                        return false;
                    }
                }
                return openParamsStack.Count == 0;
            }

            //https://www.algoexpert.io/questions/sort-stack
            // O(n^2) time | O(n) space - where n is the length of the stack
            public List<int> SortStack(List<int> stack)
            {
                if (stack.Count == 0)
                {
                    return stack;
                }

                int top = stack[stack.Count - 1];
                stack.RemoveAt(stack.Count - 1);

                SortStack(stack);

                insertInSortedOrder(stack, top);

                return stack;
            }

            public void insertInSortedOrder(List<int> stack, int value)
            {
                if (stack.Count == 0 || (stack[stack.Count - 1] <= value))
                {
                    stack.Add(value);
                    return;
                }

                int top = stack[stack.Count - 1];
                stack.RemoveAt(stack.Count - 1);

                insertInSortedOrder(stack, value);

                stack.Add(top);
            }


        }

/*
2211. Count Collisions on a Road
https://leetcode.com/problems/count-collisions-on-a-road/description/
*/
        public class CountCollisionsOnRoadSol
        {
            /*
            Approach: Two Pointers;
         Time and Space Complexity
Time Complexity : O(n).
The time complexity of the given code is primarily determined by three operations:
1.	lstrip('L'): This must check each character from the left until a non-L character is found. In the worst case, all characters are 'L', having a complexity of O(n) where n is the total number of characters in the string.
2.	rstrip('R'): Similarly, this function must check each character from the right until a non-R character is encountered. This also has a worst-case complexity of O(n) when all characters are 'R'.
3.	count('S'): This operation counts the number of 'S' characters in the modified string. This takes O(m) time where m is the length of the modified string. However, since m <= n, we also consider it O(n) for the worst case.
When these operations are added together, despite being sequential and not nested, the complexity is still governed by the longest operation which is O(n).
So, the overall time complexity of the code is O(n).
Space Complexity : O(1).
The space complexity of the code is determined by the storage required for the modified string d.
•	d is a substring of the original input directions. However, it does not require additional space proportional to the input size; it uses the slices (which are views in Python) to reference parts of the original string without creating a new copy.
•	Thus, the extra space used is for a fixed number of variables which do not grow with the size of the input.
Hence, the space complexity is O(1).
   
            */
            public int TwoPointers(string directions)
            {
                // Convert the input string to a character array for easier processing.
                char[] directionChars = directions.ToCharArray();

                // Get the length of the directionChars array.
                int length = directionChars.Length;

                // Initialize pointers for left and right.
                int leftPointer = 0;
                int rightPointer = length - 1;

                // Skip all the 'L' cars from the start as they do not contribute to collisions.
                while (leftPointer < length && directionChars[leftPointer] == 'L')
                {
                    leftPointer++;
                }

                // Skip all the 'R' cars from the end as they do not contribute to collisions.
                while (rightPointer >= 0 && directionChars[rightPointer] == 'R')
                {
                    rightPointer--;
                }

                // Initialize a counter for collisions to zero.
                int collisionsCount = 0;

                // Iterate over the remaining cars between leftPointer and rightPointer.
                for (int i = leftPointer; i <= rightPointer; ++i)
                {
                    // Count only the cars that are not 'S' (since 'S' means stopped and will not collide).
                    if (directionChars[i] != 'S')
                    {
                        collisionsCount++;
                    }
                }

                // Return the total count of collisions.
                return collisionsCount;
            }
        }



    }
}