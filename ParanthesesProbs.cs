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

        /* 1249. Minimum Remove to Make Valid Parentheses
        https://leetcode.com/problems/minimum-remove-to-make-valid-parentheses/description/
         */
        public class MinRemoveToMakeValidParenSol
        {

            /* Approach 1: Using a Stack and String Builder
            Complexity Analysis
            •	Time complexity : O(n), where n is the length of the input string.
            There are 3 loops we need to analyze. We also need to check carefully for any library functions that are not constant time.
            The first loop iterates over the string, and for each character, either does nothing, pushes to a stack or adds to a set. Pushing to a stack and adding to a set are both O(1). Because we are processing each character with an O(1) operation, this overall loop is O(n).
            The second loop (hidden in library function calls for the Python code) pops each item from the stack and adds it to the set. Again, popping items from a stack is O(1), and there are at most n characters on the stack, and so it too is O(n).
            The third loop iterates over the string again, and puts characters into a StringBuilder/ list. Checking if an item is in a set and appending to the end of a String Builder or list is O(1). Again, this is O(n) overall.
            The StringBuilder.toString() method is O(n), and so is the "".join(...). So again, this operation is O(n).
            So this gives us O(4n), and we drop the 4 because it is a constant.
            •	Space complexity : O(n), where n is the length of the input string.
            We are using a stack, set, and string builder, each of which could have up to n characters in them, and so require up to O(n) space.

             */
            public string UsingStackAndStringBuilder(string inputString)
            {
                HashSet<int> indexesToRemove = new HashSet<int>();
                Stack<int> indexStack = new Stack<int>();

                for (int index = 0; index < inputString.Length; index++)
                {
                    if (inputString[index] == '(')
                    {
                        indexStack.Push(index);
                    }
                    if (inputString[index] == ')')
                    {
                        if (indexStack.Count == 0)
                        {
                            indexesToRemove.Add(index);
                        }
                        else
                        {
                            indexStack.Pop();
                        }
                    }
                }

                // Put any indexes remaining on stack into the set.
                while (indexStack.Count > 0) indexesToRemove.Add(indexStack.Pop());

                StringBuilder stringBuilder = new StringBuilder();
                for (int index = 0; index < inputString.Length; index++)
                {
                    if (!indexesToRemove.Contains(index))
                    {
                        stringBuilder.Append(inputString[index]);
                    }
                }

                return stringBuilder.ToString();
            }

            /* Approach 2: Two Pass String Builder
            Complexity Analysis
•	Time complexity : O(n), where n is the length of the input string.
We need to analyze the removeInvalidClosing function and then the outside code.
removeInvalidClosing processes each character once and optionally modifies balance and adds the character to a string builder. Adding to the end of a string builder is O(1). As there are at most n characters to process, the overall cost is O(n).
The other code makes 2 calls to removeInvalidClosing, 2 string reversals, and 1 conversion from string builder to string. These operations are O(n), and the 3 is treated as a constant so is dropped. Again, this gives us an overall cost of O(n).
Because all parts of the code are O(n), the overall time complexity is O(n).
•	Space complexity : O(n), where n is the length of the input string.
The string building still requires O(n) space. However, the constants are smaller than the previous approach, as we no longer have the set or stack.
It is impossible to do better, because the input is an immutable string, and the output must be an immutable string. Therefore, manipulating the string cannot be done in-place, and requires O(n) space to modify.

             */
            public string TwoPassStringBuilder(string inputString)
            {
                StringBuilder result = RemoveInvalidClosing(inputString, '(', ')');
                result = RemoveInvalidClosing(result.ToString().Reverse().ToString(), ')', '(');
                return result.ToString().Reverse().ToString();
            }
            private StringBuilder RemoveInvalidClosing(string inputString, char openingBracket, char closingBracket)
            {
                StringBuilder stringBuilder = new StringBuilder();
                int balance = 0;
                for (int i = 0; i < inputString.Length; i++)
                {
                    char currentChar = inputString[i];
                    if (currentChar == openingBracket)
                    {
                        balance++;
                    }
                    if (currentChar == closingBracket)
                    {
                        if (balance == 0) continue;
                        balance--;
                    }
                    stringBuilder.Append(currentChar);
                }
                return stringBuilder;
            }
            /* Approach 3: Shortened Two Pass String Builder
Complexity Analysis
•	Time complexity : O(n), where n is the length of the input string.
Same as the above approaches. We have 2 loops that are looping through up to n characters, doing an O(1) operation on each. We also have some O(n) library function calls outside of the loops.
•	Space complexity : O(n), where n is the length of the input string.
Like the previous approach, the string building requires O(n) space.

             */
            public String ShortenedTwoPassStringBuilder(String s)
            {

                // Pass 1: Remove all invalid ")"
                StringBuilder sb = new StringBuilder();
                int openSeen = 0;
                int balance = 0;
                for (int i = 0; i < s.Length; i++)
                {
                    char c = s[i];
                    if (c == '(')
                    {
                        openSeen++;
                        balance++;
                    }
                    if (c == ')')
                    {
                        if (balance == 0) continue;
                        balance--;
                    }
                    sb.Append(c);
                }

                // Pass 2: Remove the rightmost "("
                StringBuilder result = new StringBuilder();
                int openToKeep = openSeen - balance;
                for (int i = 0; i < sb.Length; i++)
                {
                    char c = sb[i];
                    if (c == '(')
                    {
                        openToKeep--;
                        if (openToKeep < 0) continue;
                    }
                    result.Append(c);
                }

                return result.ToString();
            }
        }

        /* 921. Minimum Add to Make Parentheses Valid
        https://leetcode.com/problems/minimum-add-to-make-parentheses-valid/description/?envType=company&envId=facebook&favoriteSlug=facebook-all&difficulty=MEDIUM
         */
        class MinAddToMakeValidParenSol
        {

            /* Approach: Open Bracket Counter
            Complexity Analysis
            Here, N is the number of characters in the string s.
            •	Time complexity: O(N)
            We iterate over each character in the string s once. For each character, we either increment, decrement, or compare a counter. These operations take constant time. Therefore, the overall time complexity is linear, O(N).
            •	Space complexity: O(1)
            We use only two variables, openBrackets and minAddsRequired, to count unmatched brackets. These variables require constant space, and we do not use any extra data structures that depend on the input size. Thus, the space complexity is constant.

             */
            public int OpenBracketCounter(String s)
            {
                int openBrackets = 0;
                int minAddsRequired = 0;

                foreach (char c in s)
                {
                    if (c == '(')
                    {
                        openBrackets++;
                    }
                    else
                    {
                        // If an open bracket exists, match it with the closing one
                        // If not, we need to add an open bracket.
                        if (openBrackets > 0)
                        {
                            openBrackets--;
                        }
                        else
                        {
                            minAddsRequired++;
                        }
                    }
                }

                // Add the remaining open brackets as closing brackets would be required.
                return minAddsRequired + openBrackets;
            }
        }


        /* 241. Different Ways to Add Parentheses
        https://leetcode.com/problems/different-ways-to-add-parentheses/description/
         */
        public class DiffWaysToAddParenSol
        {

            /* Approach 1: Recursion
Complexity Analysis
Let n be the the length of the input string expression.
•	Time complexity: O(n⋅2^n)
For each sub-expression, we iterate through the string to identify the operators, which takes O(n) time. However, the key aspect is the recursive combination of results from the left and right sub-expressions. The number of results grows exponentially because each sub-expression produces multiple results, and combining these results takes O(k×l), where k and l are the numbers of results from the left and right sub-problems, respectively.
There were some suggestions to model the number of results using Catalan numbers which we deemed as incorrect. Catalan numbers apply when counting distinct ways to fully parenthesize an expression or structure. In this problem, however, we're not just counting valid ways to split the expression but also calculating and combining all possible results. This introduces exponential growth in the number of possible results, not the polynomial growth typical of Catalan numbers. The number of combinations grows exponentially with the depth of recursive splitting, which means the overall complexity is driven by the exponential growth in results.
Thus, the time complexity of the algorithm is O(n⋅2^n), where the O(2^n) factor reflects the exponential growth in the number of ways to combine results from sub-expressions.
•	Space complexity: O(2^n)
The algorithm stores the intermediate results at each step. Since the total number of results can be equal to the O(2^n), the space complexity of the algorithm is O(2^n).

             */
            public List<int> UsingRecursion(string expression)
            {
                List<int> results = new List<int>();

                // Base case: if the string is empty, return an empty list
                if (expression.Length == 0) return results;

                // Base case: if the string is a single character, treat it as a number and return it
                if (expression.Length == 1)
                {
                    results.Add(int.Parse(expression));
                    return results;
                }

                // If the string has only two characters and the first character is a digit, parse it as a number
                if (expression.Length == 2 && char.IsDigit(expression[0]))
                {
                    results.Add(int.Parse(expression));
                    return results;
                }

                // Recursive case: iterate through each character
                for (int i = 0; i < expression.Length; i++)
                {
                    char currentChar = expression[i];

                    // Skip if the current character is a digit
                    if (char.IsDigit(currentChar)) continue;

                    // Split the expression into left and right parts
                    List<int> leftResults = UsingRecursion(expression.Substring(0, i));
                    List<int> rightResults = UsingRecursion(expression.Substring(i + 1));

                    // Combine results from left and right parts
                    foreach (int leftValue in leftResults)
                    {
                        foreach (int rightValue in rightResults)
                        {
                            int computedResult = 0;

                            // Perform the operation based on the current character
                            switch (currentChar)
                            {
                                case '+':
                                    computedResult = leftValue + rightValue;
                                    break;
                                case '-':
                                    computedResult = leftValue - rightValue;
                                    break;
                                case '*':
                                    computedResult = leftValue * rightValue;
                                    break;
                            }

                            results.Add(computedResult);
                        }
                    }
                }

                return results;
            }
            /* Approach 2: Memoization
            Complexity Analysis
Let n be the the length of the input string expression.
•	Time complexity: O(n⋅2^n)
The algorithm uses memoization to store the results of sub-problems, ensuring that each sub-problem is evaluated exactly once. There are at most O(n^2) possible sub-problems, as each sub-problem is defined by its start and end indices, both ranging from 0 to n−1.
Despite the efficiency gains from memoization, the time complexity is still dominated by the recursive nature of the algorithm. The recursion tree expands exponentially, with a growth factor of O(2^n).
Thus, the overall time complexity remains O(n⋅2^n).
•	Space complexity: O(n^2⋅2^n)
The space complexity is O(n^2⋅2^n), where O(n^2) comes from the memoization table storing results for all sub-problems, and O(2^n) accounts for the space required to store the exponentially growing number of results for each sub-problem. The recursion stack depth is at most O(n), which is dominated by the exponential complexity and can therefore be omitted from the overall space complexity analysis.

             */
            public IList<int> UsingMemo(string expression)
            {
                // Initialize memoization array to store results of subproblems
                IList<int>[][] memo = new List<int>[expression.Length][];
                for (int i = 0; i < expression.Length; i++)
                {
                    memo[i] = new List<int>[expression.Length];
                }

                // Solve for the entire expression
                return ComputeResults(expression, memo, 0, expression.Length - 1);
            }

            private IList<int> ComputeResults(string expression, IList<int>[][] memo, int start, int end)
            {
                // If result is already memoized, return it
                if (memo[start][end] != null)
                {
                    return memo[start][end];
                }

                List<int> results = new List<int>();

                // Base case: single digit
                if (start == end)
                {
                    results.Add(expression[start] - '0');
                    return results;
                }

                // Base case: two-digit number
                if (end - start == 1 && Char.IsDigit(expression[start]))
                {
                    int tens = expression[start] - '0';
                    int ones = expression[end] - '0';
                    results.Add(10 * tens + ones);
                    return results;
                }

                // Recursive case: split the expression at each operator
                for (int i = start; i <= end; i++)
                {
                    char currentChar = expression[i];
                    if (Char.IsDigit(currentChar))
                    {
                        continue;
                    }

                    IList<int> leftResults = ComputeResults(expression, memo, start, i - 1);
                    IList<int> rightResults = ComputeResults(expression, memo, i + 1, end);

                    // Combine results from left and right subexpressions
                    foreach (int leftValue in leftResults)
                    {
                        foreach (int rightValue in rightResults)
                        {
                            switch (currentChar)
                            {
                                case '+':
                                    results.Add(leftValue + rightValue);
                                    break;
                                case '-':
                                    results.Add(leftValue - rightValue);
                                    break;
                                case '*':
                                    results.Add(leftValue * rightValue);
                                    break;
                            }
                        }
                    }
                }

                // Memoize the result for this subproblem
                memo[start][end] = results;

                return results;
            }
            /*             Approach 3: Tabulation
            Complexity Analysis
            Let n be the the length of the input string expression.
            •	Time complexity: O(n⋅2^n)
            Similar to the memoization approach, the algorithm evaluates each sub-problem exactly once. Thus, the time complexity remains the same as Approach 2: O(n⋅2^n).
            •	Space complexity: O(n^2⋅2^n)
            The space complexity is similar to the previous approach, with one key difference: the absence of the recursive stack space.
            However, the dp table dominates the space complexity anyway, keeping the overall space complexity as O(n^2⋅2^n).

             */
            public List<int> UsingTabulation(string expression)
            {
                int length = expression.Length;
                // Create a 2D array of lists to store results of subproblems
                List<int>[,] dp = new List<int>[length, length];

                InitializeBaseCases(expression, dp);

                // Fill the dp table for all possible subexpressions
                for (int currentLength = 3; currentLength <= length; currentLength++)
                {
                    for (int start = 0; start + currentLength - 1 < length; start++)
                    {
                        int end = start + currentLength - 1;
                        ProcessSubexpression(expression, dp, start, end);
                    }
                }

                // Return the results for the entire expression
                return dp[0, length - 1];
            }

            private void InitializeBaseCases(string expression, List<int>[,] dp)
            {
                int length = expression.Length;
                // Initialize the dp array with empty lists
                for (int i = 0; i < length; i++)
                {
                    for (int j = 0; j < length; j++)
                    {
                        dp[i, j] = new List<int>();
                    }
                }

                // Handle base cases: single digits and two-digit numbers
                for (int i = 0; i < length; i++)
                {
                    if (char.IsDigit(expression[i]))
                    {
                        // Check if it's a two-digit number
                        int firstDigit = expression[i] - '0';
                        if (i + 1 < length && char.IsDigit(expression[i + 1]))
                        {
                            int secondDigit = expression[i + 1] - '0';
                            int number = firstDigit * 10 + secondDigit;
                            dp[i, i + 1].Add(number);
                        }
                        // Single digit case
                        dp[i, i].Add(firstDigit);
                    }
                }
            }

            private void ProcessSubexpression(string expression, List<int>[,] dp, int start, int end)
            {
                // Try all possible positions to split the expression
                for (int split = start; split <= end; split++)
                {
                    if (char.IsDigit(expression[split])) continue;

                    List<int> leftResults = dp[start, split - 1];
                    List<int> rightResults = dp[split + 1, end];

                    ComputeResults(expression[split], leftResults, rightResults, dp[start, end]);
                }
            }

            private void ComputeResults(char op, List<int> leftResults, List<int> rightResults, List<int> results)
            {
                // Compute results based on the operator at position 'split'
                foreach (int leftValue in leftResults)
                {
                    foreach (int rightValue in rightResults)
                    {
                        switch (op)
                        {
                            case '+':
                                results.Add(leftValue + rightValue);
                                break;
                            case '-':
                                results.Add(leftValue - rightValue);
                                break;
                            case '*':
                                results.Add(leftValue * rightValue);
                                break;
                        }
                    }
                }
            }


        }

        /* 678. Valid Parenthesis String
        https://leetcode.com/problems/valid-parenthesis-string/description/
         */
        public class CheckValidParanthesisStringSol
        {
            /* Approach 1: Top-Down Dynamic Programming - Memoization
Complexity Analysis
Let n be the length of the input string.
•	Time complexity: O(n⋅n)
The time complexity of the isValidString function can be analyzed by considering the number of unique subproblems that need to be solved. Since there are at most n⋅n unique subproblems (indexed by index and openCount), where n is the length of the input string, and each subproblem is computed only once (due to memoization), the time complexity is bounded by the number of unique subproblems. Therefore, the time complexity can be stated as O(n⋅n).
•	Space complexity: O(n⋅n)
The space complexity of the algorithm is primarily determined by two factors: the auxiliary space used for memoization and the recursion stack space. The memoization table, denoted as memo, consumes O(n⋅n) space due to its size being proportional to the square of the length of the input string. Additionally, the recursion stack space can grow up to O(n) in the worst case, constrained by the length of the input string, as each recursive call may add a frame to the stack. Therefore, the overall space complexity is the sum of these two components, resulting in O(n⋅n)+O(n), which simplifies to O(n⋅n).

             */
            public bool TopDownDPWithMemo(string inputString)
            {
                int length = inputString.Length;
                int[,] memoizationArray = new int[length, length];
                for (int row = 0; row < length; row++)
                {
                    for (int column = 0; column < length; column++)
                    {
                        memoizationArray[row, column] = -1;
                    }
                }
                return IsValidString(0, 0, inputString, memoizationArray);
            }

            private bool IsValidString(int currentIndex, int openBracketCount, string inputString, int[,] memoizationArray)
            {
                // If reached end of the string, check if all brackets are balanced
                if (currentIndex == inputString.Length)
                {
                    return (openBracketCount == 0);
                }
                // If already computed, return memoized result
                if (memoizationArray[currentIndex, openBracketCount] != -1)
                {
                    return memoizationArray[currentIndex, openBracketCount] == 1;
                }
                bool isValid = false;
                // If encountering '*', try all possibilities
                if (inputString[currentIndex] == '*')
                {
                    isValid |= IsValidString(currentIndex + 1, openBracketCount + 1, inputString, memoizationArray); // Treat '*' as '('
                    if (openBracketCount > 0)
                    {
                        isValid |= IsValidString(currentIndex + 1, openBracketCount - 1, inputString, memoizationArray); // Treat '*' as ')'
                    }
                    isValid |= IsValidString(currentIndex + 1, openBracketCount, inputString, memoizationArray); // Treat '*' as empty
                }
                else
                {
                    // Handle '(' and ')'
                    if (inputString[currentIndex] == '(')
                    {
                        isValid = IsValidString(currentIndex + 1, openBracketCount + 1, inputString, memoizationArray); // Increment count for '('
                    }
                    else if (openBracketCount > 0)
                    {
                        isValid = IsValidString(currentIndex + 1, openBracketCount - 1, inputString, memoizationArray); // Decrement count for ')'
                    }
                }

                // Memoize and return the result
                memoizationArray[currentIndex, openBracketCount] = isValid ? 1 : 0;
                return isValid;
            }

            /* Approach 2: Bottom-Up Dynamic Programming - Tabulation
Complexity Analysis
Let n be the length of the input string.
•	Time complexity: O(n⋅n)
This is due to the nested loop structure, where the outer loop iterates over each character of the string, and the inner loop iterates over all possible counts of opening brackets.
•	Space complexity: O(n⋅n)
This is primarily due to the 2D array dp, which has dimensions (n+1)⋅(n+1).

             */
            public bool BottomUpDPTabulation(string inputString)
            {
                int stringLength = inputString.Length;
                // dp[i][j] represents if the substring starting from index i is valid with j opening brackets
                bool[,] dynamicProgrammingTable = new bool[stringLength + 1, stringLength + 1];

                // base case: an empty string with 0 opening brackets is valid
                dynamicProgrammingTable[stringLength, 0] = true;

                for (int index = stringLength - 1; index >= 0; index--)
                {
                    for (int openBracketCount = 0; openBracketCount < stringLength; openBracketCount++)
                    {
                        bool isValid = false;

                        // '*' can represent '(' or ')' or '' (empty)
                        if (inputString[index] == '*')
                        {
                            isValid |= dynamicProgrammingTable[index + 1, openBracketCount + 1]; // try '*' as '('
                                                                                                 // opening brackets to use '*' as ')'
                            if (openBracketCount > 0)
                            {
                                isValid |= dynamicProgrammingTable[index + 1, openBracketCount - 1]; // try '*' as ')'
                            }
                            isValid |= dynamicProgrammingTable[index + 1, openBracketCount]; // ignore '*'
                        }
                        else
                        {
                            // If the character is not '*', it can be '(' or ')'
                            if (inputString[index] == '(')
                            {
                                isValid |= dynamicProgrammingTable[index + 1, openBracketCount + 1]; // try '('
                            }
                            else if (openBracketCount > 0)
                            {
                                isValid |= dynamicProgrammingTable[index + 1, openBracketCount - 1]; // try ')'
                            }
                        }
                        dynamicProgrammingTable[index, openBracketCount] = isValid;
                    }
                }

                return dynamicProgrammingTable[0, 0]; // check if the entire string is valid with no excess opening brackets
            }

            /* Approach 3: Using Two Stacks
Complexity Analysis
Let n be the length of the input string.
•	Time complexity: O(n)
The algorithm iterates through the entire string once, taking O(n) time. Additionally, in the worst case, it may need to traverse both the openBrackets and asterisks stacks simultaneously to check for balanced parentheses, which also takes O(n) time. Thus, the overall time complexity is O(n).
•	Space complexity: O(n)
The algorithm uses two stacks, openBrackets, and asterisks, which could potentially hold up to O(n) elements combined in the worst case. Additionally, there are a few extra variables and loop counters, which require constant space. Therefore, the overall space complexity is O(n).

             */
            public bool UsingTwoStacks(string inputString)
            {
                // Stacks to store indices of open brackets and asterisks
                Stack<int> openBrackets = new Stack<int>();
                Stack<int> asterisks = new Stack<int>();

                for (int index = 0; index < inputString.Length; index++)
                {
                    char currentChar = inputString[index];

                    // If current character is an open bracket, push its index onto the stack
                    if (currentChar == '(')
                    {
                        openBrackets.Push(index);
                    }
                    // If current character is an asterisk, push its index onto the stack
                    else if (currentChar == '*')
                    {
                        asterisks.Push(index);
                        // current character is a closing bracket ')'
                    }
                    else
                    {
                        // If there are open brackets available, use them to balance the closing bracket
                        if (openBrackets.Count > 0)
                        {
                            openBrackets.Pop();
                            // If no open brackets are available, use an asterisk to balance the closing bracket
                        }
                        else if (asterisks.Count > 0)
                        {
                            asterisks.Pop();
                        }
                        else
                        {
                            return false;
                        }
                    }
                }

                // Check if there are remaining open brackets and asterisks that can balance them
                while (openBrackets.Count > 0 && asterisks.Count > 0)
                {
                    // If an open bracket appears after an asterisk, it cannot be balanced, return false
                    if (openBrackets.Pop() > asterisks.Pop())
                    {
                        return false; // '*' before '(' which cannot be balanced.
                    }
                }

                // If all open brackets are matched and there are no unmatched open brackets left, return true
                return openBrackets.Count == 0;
            }
            /* Approach 4: Two Pointer
Complexity Analysis
Let n be the length of the input string.
•	Time complexity: O(n)
The time complexity is O(n), as we iterate through the string once.
•	Space complexity: O(1)
The space complexity is O(1), as we use a constant amount of extra space to store the openCount and closeCount variables.

             */
            public bool UsingTwoPointers(string inputString)
            {
                int openParenthesesCount = 0;
                int closeParenthesesCount = 0;
                int stringLength = inputString.Length - 1;

                // Traverse the string from both ends simultaneously
                for (int index = 0; index <= stringLength; index++)
                {
                    // Count open parentheses or asterisks
                    if (inputString[index] == '(' || inputString[index] == '*')
                    {
                        openParenthesesCount++;
                    }
                    else
                    {
                        openParenthesesCount--;
                    }

                    // Count close parentheses or asterisks
                    if (inputString[stringLength - index] == ')' || inputString[stringLength - index] == '*')
                    {
                        closeParenthesesCount++;
                    }
                    else
                    {
                        closeParenthesesCount--;
                    }

                    // If at any point open count or close count goes negative, the string is invalid
                    if (openParenthesesCount < 0 || closeParenthesesCount < 0)
                    {
                        return false;
                    }
                }

                // If open count and close count are both non-negative, the string is valid
                return true;
            }

        }







    }
}