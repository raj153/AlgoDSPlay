using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AlgoDSPlay
{
    public class ParanthesesProbs
    {
        /*
22. Generate Parentheses		
https://leetcode.com/problems/generate-parentheses/editorial/

 */
        public IList<string> GenerateParenthesis(int n)
        {
            /*
   Approach 1: Brute Force         
   Complexity Analysis
•	Time complexity: O(2^2n⋅n)
o	We are generating all possible strings of length 2n. At each character, we have two choices: choosing ( or ), which means there are a total of 22n unique strings.
o	For each string of length 2n, we need to iterate through each character to verify it is a valid combination of parentheses, which takes an average of O(n) time.
•	Space complexity: O(2^2n⋅n)
o	While we don't count the answer as part of the space complexity, for those interested, it is the nth Catalan number, which is asymptotically bounded by nn4n. Thus answer takes O(n4n) space.
Please find the explanation behind this intuition in approach 3!
You can also refer to Catalan number on Wikipedia for more information about Catalan numbers.
o	Before we dequeue the first string of length 2n from queue, it has stored 22n−1 strings of length n−1, which takes O(22n⋅n).
o	To sum up, the overall space complexity is O(22n⋅n).

            */
            IList<string> parenths = GenerateParenthesisNaive(n);

            /*
Approach 2: Backtracking, Keep Candidate Valid
Complexity Analysis
•	Time complexity: O(4^n/root n)
o	We only track the valid prefixes during the backtracking procedure. As explained in the approach 1 time complexity analysis, the total number of valid parentheses strings is O(nn4n).
Please find the explanation behind this intuition in approach 3!
You can also refer to Catalan number on Wikipedia for more information about Catalan numbers.
o	When considering each valid string, it is important to note that we use a mutable instance (StringBuilder in Java, list in Python etc.). As a result, in order to add each instance of a valid string to answer, we must first convert it to a string. This process brings an additional n factor in the time complexity.
•	Space complexity: O(n)
o	The space complexity of a recursive call depends on the maximum depth of the recursive call stack, which is 2n. As each recursive call either adds a left parenthesis or a right parenthesis, and the total number of parentheses is 2n. Therefore, at most O(n) levels of recursion will be created, and each level consumes a constant amount of space.


            
            */
            parenths = GenerateParenthesisBacktrack(n);
            /*
Approach 3: Divide and Conquer (DAC)
 Complexity Analysis
•	Time complexity: O(4^n/root n)
o	We begin by generating all valid parentheses strings of length 2, 4, ..., 2n. As shown in approach 2, 
the time complexity for generating all valid parentheses strings of length 2n is given by the expression O(n4n). Therefore, the total time complexity can be expressed T(n)=i=1∑ni4i which is asymptotically bounded by O(n4n).
•	Space complexity: O(n)
o	We don't count the answer as part of the space complexity, so the space complexity would be the maximum depth of the recursion stack. At any given time, the recursive function call stack would contain at most n function calls.
         
            
            */
            parenths = GenerateParenthesisDAC(n);

            return parenths;


        }
        public IList<string> GenerateParenthesisDAC(int n)
        {
            if (n == 0)
            {
                return new List<string> { "" };
            }

            List<string> answer = new List<string>();
            for (int leftCount = 0; leftCount < n; ++leftCount)
            {
                foreach (string leftString in GenerateParenthesisDAC(leftCount))
                {
                    foreach (string rightString in GenerateParenthesisDAC(n - 1 -
                                                                       leftCount))
                    {
                        answer.Add("(" + leftString + ")" + rightString);
                    }
                }
            }

            return answer;
        }
        public IList<string> GenerateParenthesisBacktrack(int n)
        {
            List<string> answer = new List<string>();
            Backtracking(answer, new StringBuilder(), 0, 0, n);
            return answer;
        }

        private void Backtracking(List<string> answer, StringBuilder curString,
                                  int leftCount, int rightCount, int n)
        {
            if (curString.Length == 2 * n)
            {
                answer.Add(curString.ToString());
                return;
            }

            if (leftCount < n)
            {
                curString.Append("(");
                Backtracking(answer, curString, leftCount + 1, rightCount, n);
                curString.Remove(curString.Length - 1, 1);
            }

            if (leftCount > rightCount)
            {
                curString.Append(")");
                Backtracking(answer, curString, leftCount, rightCount + 1, n);
                curString.Remove(curString.Length - 1, 1);
            }
        }
        private bool IsValid(string pString)
        {
            int leftCount = 0;
            foreach (char p in pString.ToCharArray())
            {
                if (p == '(')
                {
                    leftCount++;
                }
                else
                {
                    leftCount--;
                }

                if (leftCount < 0)
                {
                    return false;
                }
            }

            return leftCount == 0;
        }

        public IList<string> GenerateParenthesisNaive(int n)
        {
            IList<string> answer = new List<string>();
            Queue<string> queue = new Queue<string>();
            queue.Enqueue("");
            while (queue.Count != 0)
            {
                string curString = queue.Dequeue();
                if (curString.Length == 2 * n)
                {
                    if (IsValid(curString))
                    {
                        answer.Add(curString);
                    }

                    continue;
                }

                queue.Enqueue(curString + ")");
                queue.Enqueue(curString + "(");
            }

            return answer;
        }

        /*
20. Valid Parentheses 
https://leetcode.com/problems/valid-parentheses/description/	

Approach1 : Using Stacks
Complexity analysis
•	Time complexity : O(n) because we simply traverse the given string one character at a time and push and pop operations on a stack take O(1) time.
•	Space complexity : O(n) as we push all opening brackets onto the stack and in the worst case, we will end up pushing all the brackets onto the stack. e.g. ((((((((((.

        */
        private Dictionary<char, char> mappings;
        public bool IsParenValid(string s)
        {
            mappings = new Dictionary<char, char> {
            { ')', '(' }, { '}', '{' }, { ']', '[' } };
            var stack = new Stack<char>();
            foreach (var c in s)
            {
                if (mappings.ContainsKey(c))
                {
                    char topElement = stack.Count == 0 ? '#' : stack.Pop();
                    if (topElement != mappings[c])
                    {
                        return false;
                    }
                }
                else
                {
                    stack.Push(c);
                }
            }

            return stack.Count == 0;
        }

        /*
32. Longest Valid Parentheses
https://leetcode.com/problems/longest-valid-parentheses/description/

        */
        public int LongestValidParentheses(string s)
        {
            /*
Approach 1: Brute Force
Complexity Analysis
•	Time complexity: O(n^3). Generating every possible substring from a string of length n requires O(n^2). Checking validity of a string of length n requires O(n).
•	Space complexity: O(n). A stack of depth n will be required for the longest substring.
           
            */
            int maxLen = LongestValidParenthesesNaive(s);
            /*
 Approach 2: Using Dynamic Programming           
 Complexity Analysis
•	Time complexity: O(n). Single traversal of string to fill dp array is done.
•	Space complexity: O(n). dp array of size n is used.
           
            */

            maxLen = LongestValidParenthesesDP(s);
            /*            
 Approach 3: Using Stack           
 Complexity Analysis
•	Time complexity: O(n). n is the length of the given string.
•	Space complexity: O(n). The size of stack can go up to n.

            */
            maxLen = LongestValidParenthesesStack(s);

            /*
Approach 4: Without extra space/Two Pass
Complexity Analysis
•	Time complexity: O(n). Two traversals of the string.
•	Space complexity: O(1). Only two extra variables left and right are needed.

            
            */
            maxLen = LongestValidParenthesesTwoPass(s);

            return maxLen;
        }

        public int LongestValidParenthesesNaive(string s)
        {
            int maxlen = 0;
            for (int i = 0; i < s.Length; i++)
            {
                for (int j = i + 2; j <= s.Length; j += 2)
                {
                    if (IsValid(s.Substring(i, j - i)))
                    {
                        maxlen = Math.Max(maxlen, j - i);
                    }
                }
            }

            return maxlen;
            bool IsValid(string s)
            {
                Stack<char> stack = new Stack<char>();
                for (int i = 0; i < s.Length; i++)
                {
                    if (s[i] == '(')
                    {
                        stack.Push('(');
                    }
                    else if (stack.Count > 0 && stack.Peek() == '(')
                    {
                        stack.Pop();
                    }
                    else
                    {
                        return false;
                    }
                }

                return stack.Count == 0;
            }

        }
        public int LongestValidParenthesesDP(string s)
        {
            int maxans = 0;
            int[] dp = new int[s.Length];
            for (int i = 1; i < s.Length; i++)
            {
                if (s[i] == ')')
                {
                    if (s[i - 1] == '(')
                    {
                        dp[i] = (i >= 2 ? dp[i - 2] : 0) + 2;
                    }
                    else if (i - dp[i - 1] > 0 && s[i - dp[i - 1] - 1] == '(')
                    {
                        dp[i] = dp[i - 1] +
                                ((i - dp[i - 1]) >= 2 ? dp[i - dp[i - 1] - 2] : 0) +
                                2;
                    }

                    maxans = Math.Max(maxans, dp[i]);
                }
            }

            return maxans;
        }
        public int LongestValidParenthesesStack(string s)
        {
            int maxans = 0;
            Stack<int> stack = new Stack<int>();
            stack.Push(-1);
            for (int i = 0; i < s.Length; i++)
            {
                if (s[i] == '(')
                {
                    stack.Push(i);
                }
                else
                {
                    stack.Pop();
                    if (stack.Count == 0)
                    {
                        stack.Push(i);
                    }
                    else
                    {
                        maxans = Math.Max(maxans, i - stack.Peek());
                    }
                }
            }

            return maxans;
        }
        public int LongestValidParenthesesTwoPass(string s)
        {
            int left = 0, right = 0, maxlength = 0;
            for (int i = 0; i < s.Length; i++)
            {
                if (s[i] == '(')
                {
                    left++;
                }
                else
                {
                    right++;
                }

                if (left == right)
                {
                    maxlength = Math.Max(maxlength, 2 * right);
                }
                else if (right > left)
                {
                    left = right = 0;
                }
            }

            left = right = 0;
            for (int i = s.Length - 1; i >= 0; i--)
            {
                if (s[i] == '(')
                {
                    left++;
                }
                else
                {
                    right++;
                }

                if (left == right)
                {
                    maxlength = Math.Max(maxlength, 2 * left);
                }
                else if (left > right)
                {
                    left = right = 0;
                }
            }

            return maxlength;
        }

        /*
        301. Remove Invalid Parentheses
https://leetcode.com/problems/remove-invalid-parentheses/description/	
        */

        public IList<string> RemoveInvalidParentheses(string s)
        {
            /*
            Approach 1: Backtracking
Complexity analysis
•	Time Complexity : O(2^N) since in the worst case we will have only left parentheses in the expression and for every bracket we will have two options i.e. whether to remove it or consider it. Considering that the expression has N parentheses, the time complexity will be O(2^N).
•	Space Complexity : O(N) because we are resorting to a recursive solution and for a recursive solution there is always stack space used as internal function states are saved onto a stack during recursion. The maximum depth of recursion decides the stack space used. Since we process one character at a time and the base case for the recursion is when we have processed all of the characters of the expression string, the size of the stack would be O(N). Note that we are not considering the space required to store the valid expressions. We only count the intermediate space here.
            */
            IList<string> validParens = RemoveInvalidParenthesesBacktrack(s);

            /*
 Approach 2: Limited Backtracking!           
Complexity analysis
•	Time Complexity : The optimization that we have performed is simply a better form of pruning. Pruning here is something that will vary from one test case to another. In the worst case, we can have something like ((((((((( and the left_rem = len(S) and in such a case we can discard all of the characters because all are misplaced. So, in the worst case we still have 2 options per parenthesis and that gives us a complexity of O(2^N).
•	Space Complexity : The space complexity remains the same i.e. O(N) as previous solution. We have to go to a maximum recursion depth of N before hitting the base case. Note that we are not considering the space required to store the valid expressions. We only count the intermediate space here.

            */
            validParens = RemoveInvalidParenthesesLimitedBacktrack(s);

            return validParens;

        }
        private HashSet<string> validExpressions = new HashSet<string>();
        private int minimumRemoved;

        private void Reset()
        {
            this.validExpressions.Clear();
            this.minimumRemoved = int.MaxValue;
        }

        private void Recurse(
            string inputString,
            int index,
            int leftCount,
            int rightCount,
            StringBuilder expression,
            int removedCount)
        {
            // If we have reached the end of string.
            if (index == inputString.Length)
            {
                // If the current expression is valid.
                if (leftCount == rightCount)
                {
                    // If the current count of removed parentheses is <= the current minimum count
                    if (removedCount <= this.minimumRemoved)
                    {
                        // Convert StringBuilder to a String. This is an expensive operation.
                        // So we only perform this when needed.
                        string possibleAnswer = expression.ToString();

                        // If the current count beats the overall minimum we have till now
                        if (removedCount < this.minimumRemoved)
                        {
                            this.validExpressions.Clear();
                            this.minimumRemoved = removedCount;
                        }
                        this.validExpressions.Add(possibleAnswer);
                    }
                }
            }
            else
            {
                char currentCharacter = inputString[index];
                int length = expression.Length;

                // If the current character is neither an opening bracket nor a closing one,
                // simply recurse further by adding it to the expression StringBuilder
                if (currentCharacter != '(' && currentCharacter != ')')
                {
                    expression.Append(currentCharacter);
                    this.Recurse(inputString, index + 1, leftCount, rightCount, expression, removedCount);
                    expression.Remove(length, 1);
                }
                else
                {
                    // Recursion where we delete the current character and move forward
                    this.Recurse(inputString, index + 1, leftCount, rightCount, expression, removedCount + 1);
                    expression.Append(currentCharacter);

                    // If it's an opening parenthesis, consider it and recurse
                    if (currentCharacter == '(')
                    {
                        this.Recurse(inputString, index + 1, leftCount + 1, rightCount, expression, removedCount);
                    }
                    else if (rightCount < leftCount)
                    {
                        // For a closing parenthesis, only recurse if right < left
                        this.Recurse(inputString, index + 1, leftCount, rightCount + 1, expression, removedCount);
                    }

                    // Undoing the append operation for other recursions.
                    expression.Remove(length, 1);
                }
            }
        }

        public List<string> RemoveInvalidParenthesesBacktrack(string inputString)
        {
            this.Reset();
            this.Recurse(inputString, 0, 0, 0, new StringBuilder(), 0);
            return new List<string>(this.validExpressions);
        }

        private void Recurse(
     string inputString,
     int currentIndex,
     int leftCount,
     int rightCount,
     int leftRem,
     int rightRem,
     StringBuilder expression)
        {

            // If we reached the end of the string, just check if the resulting expression is
            // valid or not and also if we have removed the total number of left and right
            // parentheses that we should have removed.
            if (currentIndex == inputString.Length)
            {
                if (leftRem == 0 && rightRem == 0)
                {
                    this.validExpressions.Add(expression.ToString());
                }
            }
            else
            {
                char currentCharacter = inputString[currentIndex];
                int expressionLength = expression.Length;

                // The discard case. Note that here we have our pruning condition.
                // We don't recurse if the remaining count for that parenthesis is == 0.
                if ((currentCharacter == '(' && leftRem > 0) || (currentCharacter == ')' && rightRem > 0))
                {
                    this.Recurse(
                        inputString,
                        currentIndex + 1,
                        leftCount,
                        rightCount,
                        leftRem - (currentCharacter == '(' ? 1 : 0),
                        rightRem - (currentCharacter == ')' ? 1 : 0),
                        expression);
                }

                expression.Append(currentCharacter);

                // Simply recurse one step further if the current character is not a parenthesis.
                if (currentCharacter != '(' && currentCharacter != ')')
                {
                    this.Recurse(inputString, currentIndex + 1, leftCount, rightCount, leftRem, rightRem, expression);
                }
                else if (currentCharacter == '(')
                {
                    // Consider an opening bracket.
                    this.Recurse(inputString, currentIndex + 1, leftCount + 1, rightCount, leftRem, rightRem, expression);
                }
                else if (rightCount < leftCount)
                {
                    // Consider a closing bracket.
                    this.Recurse(inputString, currentIndex + 1, leftCount, rightCount + 1, leftRem, rightRem, expression);
                }

                // Delete for backtracking.
                expression.Remove(expressionLength - 1, 1);
            }
        }

        public List<string> RemoveInvalidParenthesesLimitedBacktrack(string inputString)
        {
            int left = 0, right = 0;

            // First, we find out the number of misplaced left and right parentheses.
            for (int i = 0; i < inputString.Length; i++)
            {
                // Simply record the left one.
                if (inputString[i] == '(')
                {
                    left++;
                }
                else if (inputString[i] == ')')
                {
                    // If we don't have a matching left, then this is a misplaced right, record it.
                    right = left == 0 ? right + 1 : right;

                    // Decrement count of left parentheses because we have found a right
                    // which CAN be a matching one for a left.
                    left = left > 0 ? left - 1 : left;
                }
            }

            this.Recurse(inputString, 0, 0, 0, left, right, new StringBuilder());
            return new List<string>(this.validExpressions);
        }
    }
}