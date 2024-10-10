using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

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



























































    }
}