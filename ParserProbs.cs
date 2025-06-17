using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;
using AlgoDSPlay.DataStructures;

namespace AlgoDSPlay
{
    public class ParserProbs
    {


        /* 1106. Parsing A Boolean Expression
        https://leetcode.com/problems/parsing-a-boolean-expression/description/
         */

        public class ParseBoolExprSol
        {
            /* Method 1: Iterative version - Use Stack and Set.
            Time & space: O(n), n = expression.length().
             */
            public bool UsingStackAndSetIteratives(string expression)
            {
                Stack<char> stack = new Stack<char>();
                for (int index = 0; index < expression.Length; ++index)
                {
                    char currentCharacter = expression[index];
                    if (currentCharacter == ')')
                    {
                        HashSet<char> seenCharacters = new HashSet<char>();
                        while (stack.Peek() != '(')
                            seenCharacters.Add(stack.Pop());
                        stack.Pop(); // pop out '('.
                        char operatorCharacter = stack.Pop(); // get operator for current expression.
                        if (operatorCharacter == '&')
                        {
                            stack.Push(seenCharacters.Contains('f') ? 'f' : 't'); // if there is any 'f', & expression results to 'f'
                        }
                        else if (operatorCharacter == '|')
                        {
                            stack.Push(seenCharacters.Contains('t') ? 't' : 'f'); // if there is any 't', | expression results to 't'
                        }
                        else
                        { // ! expression.
                            stack.Push(seenCharacters.Contains('t') ? 'f' : 't'); // Logical NOT flips the expression.
                        }
                    }
                    else if (currentCharacter != ',')
                    {
                        stack.Push(currentCharacter);
                    }
                }
                return stack.Pop() == 't';
            }
            /* Method 2: Recursive version 
            Time & space: O(n), n = expression.length().


            */
            public bool UsingRecursion(string expression)
            {
                return Parse(expression, 0, expression.Length);
            }

            private bool Parse(string s, int lo, int hi)
            {
                char currentChar = s[lo];
                if (hi - lo == 1) return currentChar == 't'; // base case.
                bool result = currentChar == '&'; // only when currentChar is &, set result to true; otherwise false.
                for (int i = lo + 2, start = i, level = 0; i < hi; ++i)
                {
                    char nextChar = s[i];
                    if (level == 0 && (nextChar == ',' || nextChar == ')'))
                    { // locate a valid sub-expression. 
                        bool currentResult = Parse(s, start, i); // recurse to sub-problem.
                        start = i + 1; // next sub-expression start index.
                        if (currentChar == '&') result &= currentResult;
                        else if (currentChar == '|') result |= currentResult;
                        else result = !currentResult; // currentChar == '!'.
                    }
                    if (nextChar == '(') ++level;
                    if (nextChar == ')') --level;
                }
                return result;
            }

        }


        /* 736. Parse Lisp Expression
        https://leetcode.com/problems/parse-lisp-expression/description/
        https://algo.monster/liteproblems/736
         */
        class ParseLispExpressionSol
        {
            private int currentIndex;
            private string expression;
            private Dictionary<string, Stack<int>> scopes = new Dictionary<string, Stack<int>>();

            /* Time and Space Complexity
            The given code implements a parsing and evaluation of a simple expression language that supports integers, variables, and two operations: addition and multiplication, using nested expressions.
            Time Complexity:
            To determine the time complexity, we need to consider the operations performed by the evaluate function, as well as the helper functions parseVar, parseInt, and eval.
            •	The function parseVar runs in O(v) time, where v is the length of the variable name.
            •	The function parseInt runs in O(d) time, where d is the number of digits in the number.
            •	The recursive function eval is called for each subexpression and each token in the input string. In the worst case, every character can be part of a subexpression (for simple expressions, without nested ones), which would take O(n) time to parse, where n is the length of the whole expression.
            Since the eval function invokes parseVar or parseInt for each token, and in the worst-case scenario, each character could be a separate token, the overall time complexity would be O(n) for parsing. However, due to the potential for nested expressions, we have to consider that each subexpression could be recursively evaluated. Considering the nesting, the overall time complexity could be O(n * m), where m is the depth of nesting in the expression.
            Under the assumption that the recursion depth is not overly deep, which could be considered O(1) if we assume a limitation on the expression complexity, the overall time would remain O(n) for parsing and evaluating the expression. But, for complete correctness, the time complexity should be considered as O(n * m).
            Space Complexity:
            Now, let's analyze the space complexity of the code:
            •	The scope dictionary may hold a stack for each variable. In the worst case, where each variable is different, this takes O(u) space, where u is the number of unique variables in the expression.
            •	Considering the call stack depth, the maximum depth of recursive calls is determined by the depth of the nested expressions, which gives us O(m) space complexity due to recursion.
            Therefore, the space complexity is O(u + m) where u is the number of unique variables and m is the maximum depth of the nested expressions.
             */
            public int Evaluate(string expression)
            {
                this.expression = expression;
                return EvaluateExpression();
            }

            private int EvaluateExpression()
            {
                char currentChar = expression[currentIndex];
                // If not starting with '(', evaluate variable or integer.
                if (currentChar != '(')
                {
                    // If it's a variable, return its last value. Else, parse and return the integer.
                    return char.IsLower(currentChar)
                        ? scopes[ParseVariable()].Peek()
                        : ParseInteger();
                }
                // Skip past the opening parenthesis.
                currentIndex++;
                currentChar = expression[currentIndex];
                int result = 0;

                // Check if it is a 'let' expression.
                if (currentChar == 'l')
                {
                    // Move past "let ".
                    currentIndex += 4;
                    List<string> variables = new List<string>();

                    while (true)
                    {
                        string variable = ParseVariable();

                        // If we reach the end of the expression, return the last variable's value.
                        if (expression[currentIndex] == ')')
                        {
                            result = scopes[variable].Peek();
                            break;
                        }

                        variables.Add(variable);
                        currentIndex++;
                        if (!scopes.ContainsKey(variable))
                        {
                            scopes[variable] = new Stack<int>();
                        }
                        scopes[variable].Push(EvaluateExpression());
                        currentIndex++;

                        // If next character is not a variable, it's an expression to evaluate.
                        if (!char.IsLower(expression[currentIndex]))
                        {
                            result = EvaluateExpression();
                            break;
                        }
                    }

                    // Clean up the scope by removing local variable values.
                    foreach (string v in variables)
                    {
                        scopes[v].Pop();
                    }

                }
                else
                {
                    // If it's 'add' or 'mult'.
                    bool isAddition = currentChar == 'a';
                    // Move past "add " or "mult ".
                    currentIndex += isAddition ? 4 : 5;

                    // Evaluate the first and second expressions.
                    int firstExpression = EvaluateExpression();
                    currentIndex++;
                    int secondExpression = EvaluateExpression();

                    result = isAddition ? firstExpression + secondExpression : firstExpression * secondExpression;
                }
                currentIndex++; // Skip past the closing parenthesis.
                return result;
            }

            private string ParseVariable()
            {
                int startIndex = currentIndex;
                // Move past variable name.
                while (currentIndex < expression.Length
                       && expression[currentIndex] != ' '
                       && expression[currentIndex] != ')')
                {
                    currentIndex++;
                }
                // Return the variable string.
                return expression.Substring(startIndex, currentIndex - startIndex);
            }

            private int ParseInteger()
            {
                int sign = 1;
                // Check for and handle a negative sign if present.
                if (expression[currentIndex] == '-')
                {
                    sign = -1;
                    currentIndex++;
                }

                int value = 0;
                // Parse integer by multiplying previous value by 10 and adding the next digit.
                while (currentIndex < expression.Length && char.IsDigit(expression[currentIndex]))
                {
                    value = value * 10 + (expression[currentIndex] - '0');
                    currentIndex++;
                }

                // Apply sign and return result.
                return sign * value;
            }
        }

        /* 592. Fraction Addition and Subtraction
        https://leetcode.com/problems/fraction-addition-and-subtraction/description/
         */
        class FractionAdditionAndSubtractionSol
        {
            /* Approach 1: Manual Parsing + Common Denominator

            Complexity Analysis
            •	Time Complexity: O(n)
            The loop to parse through expression runs O(n) times. Inside the loop, the math operations to combine fractions and find a common denominator is done in O(1) time. Thus, the loop in total takes O(n) time.
            The FindGCD function uses Euclid's algorithm, which runs in log(min(a,b)) time.
            Thus, the total time complexity is O(n).
            •	Space Complexity: O(log(min(a,b)))
            The space complexity is determined by the recursive overhead from the FindGCD algorithm. The max depth of the call stack would be O(log(min(a,b))). Thus, the total space complexity is O(log(min(a,b))).

             */
            public String UsingManualParsingAndCommonDenom(String expression)
            {
                int num = 0;
                int denom = 1;

                int i = 0;
                while (i < expression.Length)
                {
                    int currNum = 0;
                    int currDenom = 0;

                    bool isNegative = false;

                    // check for sign
                    if (expression[i] == '-' || expression[i] == '+')
                    {
                        if (expression[i] == '-')
                        {
                            isNegative = true;
                        }
                        // move to next character
                        i++;
                    }

                    // build numerator
                    while (Char.IsDigit(expression[i]))
                    {
                        int val = expression[i] - '0';
                        currNum = currNum * 10 + val;
                        i++;
                    }

                    if (isNegative) currNum *= -1;

                    //skip divisor
                    i++;

                    // build denominator
                    while (
                        i < expression.Length &&
                        Char.IsDigit(expression[i])
                    )
                    {
                        int val = expression[i] - '0';
                        currDenom = currDenom * 10 + val;
                        i++;
                    }

                    // add fractions together using common denominator
                    num = num * currDenom + currNum * denom;
                    denom = denom * currDenom;
                }

                int gcd = Math.Abs(FindGCD(num, denom));

                // reduce fractions
                num /= gcd;
                denom /= gcd;

                return num + "/" + denom;
            }

            private int FindGCD(int a, int b)
            {
                if (a == 0) return b;
                return FindGCD(b % a, a);
            }


            /* Approach 2 - Parsing with Regular Expressions

Complexity Analysis
•	Time Complexity: O(n)
The regex parsing will take O(n) time. Processing the nums array and performing the fraction math will take a total of O(n) time as well. The FindGCD function runs in log(min(a,b)) time.
Thus, the total time complexity is O(n).
•	Space Complexity: O(log(min(a,b)))
Like before, the space complexity is determined by the recursive overhead from the FindGCD algorithm. The max depth of the call stack would be O(log(min(a,b))). Thus, the total space complexity is O(log(min(a,b))).

             */
            public string UsingParsingAndRegEx(string expression)
            {
                int num = 0;
                int denom = 1;

                // separate expression into signed numbers
                string[] nums = expression.Split(new[] { "/", "(?=[-+])" }, StringSplitOptions.RemoveEmptyEntries);

                for (int i = 0; i < nums.Length; i += 2)
                {
                    int currNum = int.Parse(nums[i]);
                    int currDenom = int.Parse(nums[i + 1]);

                    num = num * currDenom + currNum * denom;
                    denom = denom * currDenom;
                }

                int gcd = Math.Abs(FindGCD(num, denom));

                num /= gcd;
                denom /= gcd;

                return num + "/" + denom;
            }



        }


        /* 399. Evaluate Division
        https://leetcode.com/problems/evaluate-division/description/
         */
        class CalculateEquationSol
        {
            /* Approach 1: Path Search in Graph
            Complexity Analysis
Let N be the number of input equations and M be the number of queries.
•	Time Complexity: O(M⋅N)
o	First of all, we iterate through the equations to build a graph. Each equation takes O(1) time to process.
Therefore, this step will take O(N) time in total.
o	For each query, we need to traverse the graph. In the worst case, we might need to traverse the entire graph, which could take O(N).
Hence, in total, the evaluation of queries could take M⋅O(N)=O(M⋅N).
o	To sum up, the overall time complexity of the algorithm is O(N)+O(M⋅N)=O(M⋅N)
•	Space Complexity: O(N)
o	We build a graph out the equations. In the worst case where there is no overlapping among the equations, we would have N edges and 2N nodes in the graph.
Therefore, the sapce complexity of the graph is O(N+2N)=O(3N)=O(N).
o	Since we employ the recursion in the backtracking, we would consume additional memory in the function call stack, which could amount to O(N) space.
o	In addition, we used a set visited to keep track of the nodes we visited during the backtracking.
The space complexity of the visited set would be O(N).
o	To sum up, the overall space complexity of the algorithm is O(N)+O(N)+O(N)=O(N).
o	Note that we did not take into account the space needed to hold the results. Otherwise, the total space complexity would be O(N+M).

             */
            public double[] UsingPathSearchInGraph(IList<IList<string>> equations, double[] values, IList<IList<string>> queries)
            {
                Dictionary<string, Dictionary<string, double>> graph = new Dictionary<string, Dictionary<string, double>>();

                // Step 1). build the graph from the equations
                for (int i = 0; i < equations.Count; i++)
                {
                    IList<string> equation = equations[i];
                    string dividend = equation[0], divisor = equation[1];
                    double quotient = values[i];

                    if (!graph.ContainsKey(dividend))
                        graph[dividend] = new Dictionary<string, double>();
                    if (!graph.ContainsKey(divisor))
                        graph[divisor] = new Dictionary<string, double>();

                    graph[dividend][divisor] = quotient;
                    graph[divisor][dividend] = 1 / quotient;
                }

                // Step 2). Evaluate each query via backtracking (DFS)
                double[] results = new double[queries.Count];
                for (int i = 0; i < queries.Count; i++)
                {
                    IList<string> query = queries[i];
                    string dividend = query[0], divisor = query[1];

                    if (!graph.ContainsKey(dividend) || !graph.ContainsKey(divisor))
                        results[i] = -1.0;
                    else if (dividend == divisor)
                        results[i] = 1.0;
                    else
                    {
                        HashSet<string> visited = new HashSet<string>();
                        results[i] = BacktrackEvaluate(graph, dividend, divisor, 1, visited);
                    }
                }

                return results;
            }

            private double BacktrackEvaluate(Dictionary<string, Dictionary<string, double>> graph, string currentNode, string targetNode, double accumulatedProduct, HashSet<string> visited)
            {
                // mark the visit
                visited.Add(currentNode);
                double result = -1.0;

                Dictionary<string, double> neighbors = graph[currentNode];
                if (neighbors.ContainsKey(targetNode))
                    result = accumulatedProduct * neighbors[targetNode];
                else
                {
                    foreach (KeyValuePair<string, double> pair in neighbors)
                    {
                        string nextNode = pair.Key;
                        if (visited.Contains(nextNode))
                            continue;
                        result = BacktrackEvaluate(graph, nextNode, targetNode, accumulatedProduct * pair.Value, visited);
                        if (result != -1.0)
                            break;
                    }
                }

                // unmark the visit, for the next backtracking
                visited.Remove(currentNode);
                return result;
            }
            /* Approach 2: Union-Find with Weights
Complexity Analysis
Since we applied the Union-Find data structure in our algorithm, we would like to start with a statement on the time complexity of the data structure, as follows:
Statement: If M operations, either Union or Find, are applied to N elements, the total run time is O(M⋅logN), when we perform unions arbitrarily instead of by size or rank.
One can refer to the Wikipedia for more details.
In our case, the maximum number of elements in the Union-Find data structure is equal to twice of the number of equations, i.e. each equation introduces two new variables.
Let N be the number of input equations and M be the number of queries.
•	Time Complexity: O((M+N)⋅logN).
o	First of all, we iterate through each input equation and invoke union() upon it. As a result, the overall time complexity of this step is O(N⋅logN).
o	As the second step, we then evaluate the query one by one. For each evaluation, we invoke the find() function at most twice. Therefore, the overall time complexity of this step is O(M⋅logN).
o	To sum up, the total time complexity of the algorithm is O((M+N)⋅logN).
o	Note, as compared to the DFS/BFS search approach, Union-Find data structure is more efficient for the repetitive/redundant query scenario.
o	Once we evaluate a query with Union-Find, all the subsequent repetitive queries or any query that has the overlapping with the previous query in terms of variable group could be evaluated in constant time.
For instance, in the above example, once the query of ca is evaluated, if later we want to evaluate ba, we could instantly obtain the states of variables a and b without triggering the chain update again.
While for DFS/BFS approaches, the cost of evaluating each query is independent for each other.
•	Space Complexity: O(N)
o	The space complexity of our Union-Find data structure is O(N) where we maintain a state for each variable.
o	Since the find() function is implemented with recursion, there would be some additional memory consumption in function call stack, which could amount to O(N).
o	To sum up, the total space complexity of the algorithm is O(N)+O(N)=O(N).
o	Again, we did not take into account the space needed to hold the results. Otherwise, the total space complexity would be O(N+M).

             */
            public double[] UsingUnionFindWithWeights(IList<IList<string>> equations, double[] values, IList<IList<string>> queries)
            {
                Dictionary<string, Tuple<string, double>> gidWeight = new Dictionary<string, Tuple<string, double>>();

                // Step 1). build the union groups
                for (int i = 0; i < equations.Count; i++)
                {
                    IList<string> equation = equations[i];
                    string dividend = equation[0], divisor = equation[1];
                    double quotient = values[i];

                    Union(gidWeight, dividend, divisor, quotient);
                }

                // Step 2). run the evaluation, with "lazy" updates in Find() function
                double[] results = new double[queries.Count];
                for (int i = 0; i < queries.Count; i++)
                {
                    IList<string> query = queries[i];
                    string dividend = query[0], divisor = query[1];

                    if (!gidWeight.ContainsKey(dividend) || !gidWeight.ContainsKey(divisor))
                        // case 1). at least one variable did not appear before
                        results[i] = -1.0;
                    else
                    {
                        Tuple<string, double> dividendEntry = Find(gidWeight, dividend);
                        Tuple<string, double> divisorEntry = Find(gidWeight, divisor);

                        string dividendGid = dividendEntry.Item1;
                        string divisorGid = divisorEntry.Item1;
                        double dividendWeight = dividendEntry.Item2;
                        double divisorWeight = divisorEntry.Item2;

                        if (!dividendGid.Equals(divisorGid))
                            // case 2). the variables do not belong to the same chain/group
                            results[i] = -1.0;
                        else
                            // case 3). there is a chain/path between the variables
                            results[i] = dividendWeight / divisorWeight;
                    }
                }

                return results;
            }

            private Tuple<string, double> Find(Dictionary<string, Tuple<string, double>> gidWeight, string nodeId)
            {
                if (!gidWeight.ContainsKey(nodeId))
                    gidWeight[nodeId] = new Tuple<string, double>(nodeId, 1.0);

                Tuple<string, double> entry = gidWeight[nodeId];
                // found inconsistency, trigger chain update
                if (!entry.Item1.Equals(nodeId))
                {
                    Tuple<string, double> newEntry = Find(gidWeight, entry.Item1);
                    gidWeight[nodeId] = new Tuple<string, double>(
                        newEntry.Item1, entry.Item2 * newEntry.Item2);
                }

                return gidWeight[nodeId];
            }

            private void Union(Dictionary<string, Tuple<string, double>> gidWeight, string dividend, string divisor, double value)
            {
                Tuple<string, double> dividendEntry = Find(gidWeight, dividend);
                Tuple<string, double> divisorEntry = Find(gidWeight, divisor);

                string dividendGid = dividendEntry.Item1;
                string divisorGid = divisorEntry.Item1;
                double dividendWeight = dividendEntry.Item2;
                double divisorWeight = divisorEntry.Item2;

                // merge the two groups together,
                // by attaching the dividend group to the one of divisor
                if (!dividendGid.Equals(divisorGid))
                {
                    gidWeight[dividendGid] = new Tuple<string, double>(divisorGid,
                        divisorWeight * value / dividendWeight);
                }
            }


        }

        /* 
        1410. HTML Entity Parser
        https://leetcode.com/problems/html-entity-parser/description/
        https://algo.monster/liteproblems/1410
         */

        public class HTMLEntityParserSol
        {
            /* Time and Space Complexity
The time complexity of the algorithm is O(n), where n is the length of the input string text. This is because, in the worst case, each character in the string is visited once. The check text[i:j] in d is O(1) because dictionary lookups in Python are constant time, and the loop for range l doesn't significantly affect the time complexity as it's bounded (maximum length of the strings in the dictionary d is 7, which is a constant).
The space complexity of the code is O(n) as well, where n is the length of the text. This is because a list ans is being used to construct the output, and it can grow up to the length of the input string in the case where no entity is replaced.
 */
            public string EntityParser(string inputText)
            {
                // Create a dictionary for the HTML entity to its corresponding character.
                Dictionary<string, string> htmlEntityMap = new Dictionary<string, string>
        {
            { "&quot;", "\"" },
            { "&apos;", "'" },
            { "&amp;", "&" },
            { "&gt;", ">" },
            { "&lt;", "<" },
            { "&frasl;", "/" }
        };

                // StringBuilder to hold the final parsed string.
                StringBuilder parsedString = new StringBuilder();

                // Variable to track the current index in the input string.
                int currentIndex = 0;

                // Length of the input text.
                int textLength = inputText.Length;

                // Iterate over the input text to find and replace entities.
                while (currentIndex < textLength)
                {
                    // Flag to mark if we found an entity match.
                    bool isEntityFound = false;

                    // Try all possible lengths for an entity (entities are between 1 and 7 characters long).
                    for (int length = 1; length <= 7 && currentIndex + length <= textLength; ++length)
                    {
                        // Extract a substring of the current length from the current index.
                        string currentSubstring = inputText.Substring(currentIndex, length);

                        // Check if the current substring is a known entity.
                        if (htmlEntityMap.ContainsKey(currentSubstring))
                        {
                            // If it's an entity, append the corresponding character to parsedString.
                            parsedString.Append(htmlEntityMap[currentSubstring]);

                            // Move the current index forward past the entity.
                            currentIndex += length;

                            // Indicate that we found and handled an entity.
                            isEntityFound = true;

                            // Break out of the loop as we only want to match the longest possible entity.
                            break;
                        }
                    }

                    // If no entity was found, append the current character to parsedString and move to the next character.
                    if (!isEntityFound)
                    {
                        parsedString.Append(inputText[currentIndex++]);
                    }
                }

                // Return the fully parsed string with all entities replaced.
                return parsedString.ToString();
            }
        }

        /* 439. Ternary Expression Parser
        https://leetcode.com/problems/ternary-expression-parser/description/
         */

        public class ParseTernarySol
        {
            /* Approach 1: Find Rightmost Atomic Expression
            Complexity Analysis
Let N be the length of expression.
•	Time complexity: O(N^2).
The helper function isValidAtomic(s) takes O(1) time. The helper function solveAtomic(s) takes O(1) time.
We are reducing the length of expression by 4 in each iteration. Thus, the number of iterations will be N/4. In each iteration,
o	we are finding the rightmost valid atomic expression. This takes O(N) time.
o	Then we are re-building the expression. This takes O(N) time.
o	Thus, time complexity of each iteration is O(2N)=O(N).
Hence, there will be O(N) iterations each taking O(N) time. Thus, the total time complexity will be O(N^2).
•	Space complexity: O(N).
We are not using any extra space. Thus, the space complexity will be O(1) in languages where string is mutable. However, we are modifying the input which may not be considered a good practice. If we created a copy of the input and performed operations on that, we would have O(N) space.
Also, if the string is immutable in the language, then the space complexity will be O(N), because for re-building the expression, we will be creating a new string of length N.

             */
            public string UsingFindRightMostAtomicExpression(string expression)
            {
                // Checks if the string expression is a valid atomic expression
                Func<string, bool> isValidAtomic = s => (s[0] == 'T' || s[0] == 'F') && s[1] == '?' &&
                    (char.IsDigit(s[2]) || s[2] == 'T' || s[2] == 'F') && s[3] == ':' &&
                    (char.IsDigit(s[4]) || s[4] == 'T' || s[4] == 'F');

                // Returns the value of the atomic expression
                Func<string, string> solveAtomic = s => s[0] == 'T' ? s.Substring(2, 1) : s.Substring(4, 1);

                // Reduce expression until we are left with a single character
                while (expression.Length != 1)
                {
                    int j = expression.Length - 1;
                    while (!isValidAtomic(expression.Substring(j - 4, 5)))
                    {
                        j--;
                    }
                    expression = expression.Substring(0, j - 4) + solveAtomic(expression.Substring(j - 4, 5)) + expression.Substring(j + 1);
                }

                // Return the final character
                return expression;
            }
            /* Approach 2: Reverse Polish Notation
Complexity Analysis
Let N be the length of expression.
•	Time complexity: O(N2).
We are reducing the length of expression by 4 in each iteration. Thus, the number of iterations will be N/4. In each iteration,
o	we are finding the index of ?. This takes O(N) time.
o	Then we are re-building the expression. This takes O(N) time.
o	Thus, time complexity of each iteration is O(2N)=O(N).
Hence, there will be O(N) iterations each taking O(N) time. Thus, the total time complexity will be O(N2).
•	Space complexity: O(N).
We are not using any extra space. Thus, the space complexity will be O(1) in languages where string is mutable. However, we are modifying the input which may not be considered a good practice. If we created a copy of the input and performed operations on that, we would have O(N) space.
Also, if the string is immutable in the language, then the space complexity will be O(N), because for re-building the expression, we will be creating a new string of length N.

             */
            public string UsingReversePolishNotation(string expression)
            {
                // Reduce expression until we are left with a single character
                while (expression.Length != 1)
                {
                    int questionMarkIndex = expression.Length - 1;
                    while (expression[questionMarkIndex] != '?')
                    {
                        questionMarkIndex--;
                    }

                    // Find the value of the expression.
                    char value;
                    if (expression[questionMarkIndex - 1] == 'T')
                    {
                        value = expression[questionMarkIndex + 1];
                    }
                    else
                    {
                        value = expression[questionMarkIndex + 3];
                    }

                    // Replace the expression with the value
                    expression = expression.Substring(0, questionMarkIndex - 1) + value + expression.Substring(questionMarkIndex + 4);
                }

                // Return the final character
                return expression;
            }
            /* Approach 3: Reverse Polish Notation using Stack
            Complexity Analysis
Let N be the length of expression.
•	Time complexity: O(N).
We are processing each character only once. Thus, the time complexity will be O(N). In every iteration, we are pushing and popping from the stack. This takes O(1) time. Thus, the total time complexity will be O(N).
•	Space complexity: O(N).
We are using a stack of size O(N). Thus, the space complexity will be O(N).

             */
            public string UsingReversePolishNotationWithStack(string expression)
            {

                // Initialize a stack
                Stack<char> stack = new Stack<char>();
                int i = expression.Length - 1;

                // Traverse the expression from right to left
                while (i >= 0)
                {

                    // Current character
                    char curr = expression[i];

                    // Push every T, F, and digit on the stack
                    if (curr >= '0' && curr <= '9' || curr == 'T' || curr == 'F')
                    {
                        stack.Push(curr);
                    }

                    // As soon as we encounter ?, 
                    // replace top two elements of the stack with one
                    else if (curr == '?')
                    {
                        char onTrue = stack.Pop();
                        char onFalse = stack.Pop();
                        stack.Push(expression[i - 1] == 'T' ? onTrue : onFalse);

                        // Decrement i by 1 as we have already used
                        // Previous Boolean character
                        i--;
                    }

                    // Go to the previous character
                    i--;
                }

                // Return the final character
                return stack.Pop().ToString();
            }
            /* Approach 4: Binary Tree
Complexity Analysis
Let N be the length of expression, and H be the height of the binary tree constructed from expression.
•	Time complexity: O(N).
Constructing the binary tree takes O(N) time. Parsing the binary tree takes O(H) time. Since H≤N, the total time complexity will be O(N).
•	Space complexity: O(N).
For constructing the binary tree, we are saving N nodes. Thus, the space complexity will be O(N).

             */
            int index = 0;

            public String UsingBinaryTree(String expression)
            {

                // Construct Binary Tree
                TreeNode root = ConstructTree(expression);

                // Parse the binary tree till we reach the leaf node
                while (root.Left != null && root.Right != null)
                {
                    if (root.Val == 'T')
                    {
                        root = root.Left;
                    }
                    else
                    {
                        root = root.Right;
                    }
                }

                return root.Val.ToString();
            }

            private TreeNode ConstructTree(String expression)
            {

                // Storing current character of expression
                TreeNode root = new TreeNode(expression[index]);

                // If last character of expression, return
                if (index == expression.Length - 1)
                {
                    return root;
                }

                // Check next character
                index++;
                if (expression[index] == '?')
                {
                    index++;
                    root.Left = ConstructTree(expression);
                    index++;
                    root.Right = ConstructTree(expression);
                }

                return root;
            }

            /* Approach 5: Recursion
            Complexity Analysis
Let N be the length of expression.
•	Time complexity: O(N2).
In worst case, when expression is of the form
T ? T ? .......... : D : D
where D is a single character, we will have to traverse almost the entire expression to find the corresponding :. This may have to do 2N times.
Thus, the time complexity will be O(N2).
•	Space complexity: O(N).
We are using a recursion stack. The maximum depth of the recursion stack will be O(N). Thus, the space complexity will be O(N).
Note : We can reduce the time complexity of this approach to O(N) by modifying implementation a bit. The major bottleneck here was finding the corresponding :. Readers can ponder and come up with their interesting O(N) recursive implementation in the comments section.

             */
            public string UsingRecursion(string expression)
            {
                return Solve(expression, 0, expression.Length - 1);
            }

            private string Solve(string expression, int startIndex, int endIndex)
            {

                // If expression is a single character, return it
                if (startIndex == endIndex)
                {
                    return expression.Substring(startIndex, 1);
                }

                // Find the index of ?
                int questionMarkIndex = startIndex;
                while (expression[questionMarkIndex] != '?')
                {
                    questionMarkIndex++;
                }

                // Find one index after corresponding :
                int aheadColonIndex = questionMarkIndex + 1;
                int count = 1;
                while (count != 0)
                {
                    if (expression[aheadColonIndex] == '?')
                    {
                        count++;
                    }
                    else if (expression[aheadColonIndex] == ':')
                    {
                        count--;
                    }
                    aheadColonIndex++;
                }

                // Check the value of B and recursively solve
                if (expression[startIndex] == 'T')
                {
                    return Solve(expression, questionMarkIndex + 1, aheadColonIndex - 2);
                }
                else
                {
                    return Solve(expression, aheadColonIndex, endIndex);
                }
            }

            /* Approach 6: Constant Space Solution
            Complexity Analysis
Let N be the length of expression.
•	Time complexity: O(N).
We are processing each character only once. Thus, the time complexity will be O(N). In every iteration, we are incrementing i at least by 2. Thus, the total time complexity will be O(N).
•	Space complexity: O(1).
We are not using any extra space. Thus, the space complexity will be O(1).

             */
            public string UsingSpaceOptimal(string expression)
            {
                // Pointer for Traversal. It will maintain Loop Invariant.
                int index = 0;

                // Loop invariant: We will always be at the first character of 
                // expression which we should FOCUS on.
                while (true)
                {
                    // If this first character is not boolean, it means no nesting
                    // is there. Thus, we can simply return this character.
                    if (expression[index] != 'T' && expression[index] != 'F')
                    {
                        return expression[index].ToString();
                    }

                    // If this is last character, then we can simply return this
                    if (index == expression.Length - 1)
                    {
                        return expression[index].ToString();
                    }

                    // If succeeding character is :, it means we have processed
                    // the FOCUS part. Ignore the ahead part and return this character.
                    if (expression[index + 1] == ':')
                    {
                        return expression[index].ToString();
                    }

                    // Now it means this character is boolean followed by ?.
                    // If this boolean is T, then process after ? sub-expression.
                    if (expression[index] == 'T')
                    {
                        index += 2;
                    }
                    // If this boolean is F, then make index point to the character
                    // after ": of this ?". To have corresponding :, we 
                    // can maintain count
                    else
                    {
                        int count = 1;
                        index += 2;
                        while (count != 0)
                        {
                            if (expression[index] == ':')
                            {
                                count--;
                            }
                            else if (expression[index] == '?')
                            {
                                count++;
                            }
                            index++;
                        }
                    }
                }
            }
        }


        /*        385. Mini Parser
        https://leetcode.com/problems/mini-parser/description/
        https://algo.monster/liteproblems/385
         */

        class DeserializeSol
        {

            /* Time and Space Complexity
            The time complexity of the code is O(n) where n is the length of the input string s. This is because the function involves a single loop through the input string, performing a constant amount of work for each character in the string.
            The space complexity of the code is O(n), also dependent on the length of the string s. In the worst case, the input string could represent a deeply nested list, requiring a new NestedInteger object for every [ encountered before any ] is encountered, which are stored in the stack stk. In the worst-case scenario, this stack could have as many nested NestedInteger objects as there are characters in the input string if the structure were to be very unbalanced. However, this is a very conservative estimation. In practical scenarios, the number of NestedInteger objects will often be less than n.
             */
            // Deserializes a string representation of a nested list into a NestedInteger object.
            public NestedInteger Deserialize(String s)
            {
                // If the string starts with an integer, parse it and return a NestedInteger with that value.
                if (s[0] != '[')
                {
                    return new NestedInteger(int.Parse(s));
                }

                // Initialize a stack to hold the NestedInteger objects.
                Stack<NestedInteger> stack = new();
                int number = 0; // Used to store the current number being processed.
                bool isNegative = false; // Flag to check if the current number is negative.

                // Iterate through each character in the string.
                for (int i = 0; i < s.Length; ++i)
                {
                    char character = s[i];
                    if (character == '-')
                    {
                        // If the current character is a minus sign, set the isNegative flag.
                        isNegative = true;
                    }
                    else if (Char.IsDigit(character))
                    {
                        // If the current character is a digit, add it to the current number.
                        number = number * 10 + character - '0';
                    }
                    else if (character == '[')
                    {
                        // If the current character is an open bracket, push an empty NestedInteger onto the stack.
                        stack.Push(new NestedInteger());
                    }
                    else if (character == ',' || character == ']')
                    {
                        // If the character is a comma or a close bracket,
                        // and previous character was a digit, finalize and push the number onto the stack.
                        if (Char.IsDigit(s[i - 1]))
                        {
                            if (isNegative)
                            {
                                number = -number; // Apply the negative sign if applicable.
                            }
                            stack.Peek().Add(new NestedInteger(number)); // Add the number as a NestedInteger.
                        }
                        // Reset variables for processing the next number.
                        number = 0;
                        isNegative = false;

                        // If the character is a close bracket and there is more than one NestedInteger on the stack,
                        // pop the top NestedInteger and add it to the next NestedInteger on the stack.
                        if (character == ']' && stack.Count > 1)
                        {
                            NestedInteger topNestedInteger = stack.Pop();
                            stack.Peek().Add(topNestedInteger);
                        }
                    }
                }

                // The top of the stack contains the deserialized NestedInteger.
                return stack.Peek();
            }
        }
































    }
}