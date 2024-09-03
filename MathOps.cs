using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace AlgoDSPlay
{
    public class MathOps
    {
        //https://www.algoexpert.io/questions/nth-fibonacci
        public static int GetNthFib(int n)
        {

            //1.Naive Recursion- T:O(2^n) | S:O(n)
            int result = GetNthFibNaiveRec(n);

            //2.Recursion with Memorization- T:O(n) | S:O(n)
            Dictionary<int, int> memoize = new Dictionary<int, int>();
            memoize.Add(1, 0);
            memoize.Add(2, 1);
            result = GetNthFibMemorizeRec(n, memoize);

            //3.Iterative - T:O(n) | S:O(1)
            result = GetNthFibIterative(n);
            result = GetNthFibIterative2(n);
            return result;
        }

        private static int GetNthFibIterative2(int n)
        {
            int[] lastTwo = new int[] { 0, 1 };

            int counter = 3;

            while (counter <= n)
            {
                int nextFib = lastTwo[0] + lastTwo[1];
                lastTwo[0] = lastTwo[1];
                lastTwo[1] = nextFib;
                counter++;
            }
            return n > 1 ? lastTwo[1] : lastTwo[0];
        }

        private static int GetNthFibIterative(int n)
        {
            int first = 0, second = 1;
            //0 1 1 2 3 5
            int result = 0;
            for (int i = 3; i <= n; i++)
            {
                int nextFib = first + second;
                first = second;
                second = nextFib;
                //result += nextFib;
                result = nextFib;
            }
            return result;
        }

        private static int GetNthFibMemorizeRec(int n, Dictionary<int, int> memoize)
        {
            if (memoize.ContainsKey(n))
                return memoize[n];

            memoize.Add(n, GetNthFibMemorizeRec(n - 1, memoize) + GetNthFibMemorizeRec(n - 2, memoize));
            return memoize[n];

        }

        private static int GetNthFibNaiveRec(int n)
        {
            if (n == 2) return 1;
            if (n == 1) return 0;

            return GetNthFibNaiveRec(n - 1) + GetNthFibNaiveRec(n - 2);

        }
        /*
600. Non-negative Integers without Consecutive Ones
https://leetcode.com/problems/non-negative-integers-without-consecutive-ones/description/

        */
        public int FindIntegers(int n)
        {
            /*
Approach #1 Brute Force [Time Limit Exceeded]
Complexity Analysis
•	Time complexity : O(32∗n). We test the 32 consecutive positions of every number from 0 to n. Here, n refers to given number.
•	Space complexity : O(1). Constant space is used.            
            */
            int findIntegrs = FindIntegersNaive(n);
            /*
 Approach #2 Better Brute Force [Time Limit Exceeded]           
 Complexity Analysis
•	Time complexity : O(x). Only x numbers are generated. Here, x refers to the resultant count to be returned.
•	Space complexity : O(log(max_int)=32). The depth of recursion tree can go upto 32.
           
            */
            findIntegrs = FindIntegersNaive2(n);

            /*
Approach #3 Using Bit Manipulation             
Complexity Analysis**
•	Time complexity : O(log2(max_int)=32). One loop to fill f array and one loop to check all bits of num.
•	Space complexity : O(log2(max_int)=32). f array of size 32 is used.
            */
            findIntegrs = FindIntegersBitManip(n);

            return findIntegrs;

        }
        public int FindIntegersBitManip(int num)
        {
            int[] f = new int[32];
            f[0] = 1;
            f[1] = 2;
            for (int j = 2; j < f.Length; j++)
                f[j] = f[j - 1] + f[j - 2];
            int i = 30, sum = 0, prev_bit = 0;
            while (i >= 0)
            {
                if ((num & (1 << i)) != 0)
                {
                    sum += f[i];
                    if (prev_bit == 1)
                    {
                        sum--;
                        break;
                    }
                    prev_bit = 1;
                }
                else
                    prev_bit = 0;
                i--;
            }
            return sum + 1;
        }
        public int FindIntegersNaive2(int num)
        {
            return Find(0, 0, num, false);
        }
        public int Find(int i, int sum, int num, bool prev)
        {
            if (sum > num)
                return 0;
            if (1 << i > num)
                return 1;
            if (prev)
                return Find(i + 1, sum, num, false);
            return Find(i + 1, sum, num, false) + Find(i + 1, sum + (1 << i), num, true);
        }
        public int FindIntegersNaive(int num)
        {
            int count = 0;
            for (int i = 0; i <= num; i++)
                if (Check(i))
                    count++;
            return count;
        }
        public bool Check(int n)
        {
            int i = 31;
            while (i > 0)
            {
                if ((n & (1 << i)) != 0 && (n & (1 << (i - 1))) != 0)
                    return false;
                i--;
            }
            return true;
        }

        /*
        12. Integer to Roman
https://leetcode.com/problems/integer-to-roman/description/	

        */
        public string IntToRoman(int num)
        {
            /*
Approach 1: Greedy
Complexity Analysis
•	Time complexity : O(1).
As there is a finite set of roman numerals, there is a hard upper limit on how many times the loop can iterate. This upper limit is 15 times, and it occurs for the number 3888, which has a representation of MMMDCCCLXXXVIII. Therefore, we say the time complexity is constant, i.e. O(1).
•	Space complexity : O(1).
The amount of memory used does not change with the size of the input integer, and is therefore constant.

            
            */
            string roman = IntToRomanGreedy(num);
            /*
Approach 2: Hardcode Digits (HD)
Complexity Analysis
•	Time complexity : O(1).
The same number of operations is done, regardless of the size of the input. Therefore, the time complexity is constant.
•	Space complexity : O(1).
While we have Arrays, they are the same size, regardless of the size of the input. Therefore, they are constant for the purpose of space-complexity analysis.
The downside of this approach is that it is inflexible if Roman Numerals were to be extended (which is an interesting follow-up question). For example, what if we said the symbol H now represents 5000, and P now represents 10000, allowing us to represent numbers up to 39999? Approach 1 will be a lot quicker to modify, as you simply need to add these 2 values to the code without doing any calculations. But for Approach 2, you'll need to calculate and hardcode ten new representations. What if we then added symbols to be able to go up to 399,999,999? Approach 2 becomes more and more difficult to manage, the more symbols we add.          
            
            */
            roman = IntToRomanHD(num);

            return roman;

        }
        public string IntToRomanGreedy(int num)
        {
            int[] values = { 1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1 };
            string[] symbols = { "M",  "CM", "D",  "CD", "C",  "XC", "L",
                             "XL", "X",  "IX", "V",  "IV", "I" };
            StringBuilder roman = new StringBuilder();
            // Loop through each symbol, stopping if num becomes 0.
            for (int i = 0; i < values.Length && num > 0; i++)
            {
                // Repeat while the current symbol still fits into num.
                while (values[i] <= num)
                {
                    num -= values[i];
                    roman.Append(symbols[i]);
                }
            }

            return roman.ToString();
        }
        private static readonly string[] thousands = { "", "M", "MM", "MMM" };

        private static readonly string[] hundreds = {
        "", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"
    };

        private static readonly string[] tens = { "",  "X",  "XX",  "XXX",  "XL",
                                              "L", "LX", "LXX", "LXXX", "XC" };

        private static readonly string[] ones = { "",  "I",  "II",  "III",  "IV",
                                              "V", "VI", "VII", "VIII", "IX" };

        public string IntToRomanHD(int num)
        {
            return thousands[num / 1000] + hundreds[num % 1000 / 100] +
                   tens[num % 100 / 10] + ones[num % 10];
        }

        /*
273. Integer to English Words
https://leetcode.com/problems/integer-to-english-words/	

        */
        public string NumberToWords(int num)
        {
            /*
Approach 1: Recursive Approach
Complexity Analysis
Let N be the number.
•	Time complexity: O(log N base 10)
The time complexity is O(log10N) because the number of recursive calls is proportional to the number of digits in the number, which grows logarithmically with the size of the number.
•	Space complexity: O(log N base 10)
The space complexity is O(log10N), mainly because of the recursion stack. Each recursive call adds a frame to the stack until the base case is reached, leading to space usage proportional to the number of digits in the number.

            */
            string word = NumberToWordsRec(num);
            /*
Approach 2: Iterative Approach
Complexity Analysis
Let N be the number.
•	Time complexity: O(log N base 10)
O(log10N), because the number is divided by 1000 in each iteration, making the number of iterations proportional to the number of chunks, which is logarithmic.
•	Space complexity: O(1)
O(1), constant space. The space used is independent of the number's size, as it involves only a few string builders and arrays.

            */
            word = NumberToWordsIterative(num);
            /*
Approach 3: Pair-Based Approach
Complexity Analysis
Let K be the number of pairs in numberToWordsMap and N be the number.
•	Time complexity: O(K)
The time complexity is O(K) because the loop iterates through the pairs until it finds a match. This complexity is linear with respect to the number of pairs, which is constant in practice as the number of pairs is fixed.
•	Space complexity: O(log10N)
O(log10N), mainly due to the recursion stack in the convert function. The space used is proportional to the number of recursive calls made.

            */
            word = NumberToWordsPB(num);

            return word;

        }
        // Arrays to store words for numbers less than 10, 20, and 100
        private static readonly string[] belowTen = { "", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine" };
        private static readonly string[] belowTwenty = { "Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen" };
        private static readonly string[] belowHundred = { "", "Ten", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety" };

        // Main function to convert a number to English words
        public String NumberToWordsRec(int num)
        {
            // Handle the special case where the number is zero
            if (num == 0)
            {
                return "Zero";
            }
            // Call the helper function to start the conversion
            return ConvertToWordsRec(num);
        }

        // Recursive function to convert numbers to words
        // Handles numbers based on their ranges: <10, <20, <100, <1000, <1000000, <1000000000, and >=1000000000
        private String ConvertToWordsRec(int num)
        {
            if (num < 10)
            {
                return belowTen[num];
            }
            if (num < 20)
            {
                return belowTwenty[num - 10];
            }
            if (num < 100)
            {
                return belowHundred[num / 10] + (num % 10 != 0 ? " " + ConvertToWordsRec(num % 10) : "");
            }
            if (num < 1000)
            {
                return ConvertToWordsRec(num / 100) + " Hundred" + (num % 100 != 0 ? " " + ConvertToWordsRec(num % 100) : "");
            }
            if (num < 1000000)
            {
                return ConvertToWordsRec(num / 1000) + " Thousand" + (num % 1000 != 0 ? " " + ConvertToWordsRec(num % 1000) : "");
            }
            if (num < 1000000000)
            {
                return ConvertToWordsRec(num / 1000000) + " Million" + (num % 1000000 != 0 ? " " + ConvertToWordsRec(num % 1000000) : "");
            }
            return ConvertToWordsRec(num / 1000000000) + " Billion" + (num % 1000000000 != 0 ? " " + ConvertToWordsRec(num % 1000000000) : "");
        }
        public String NumberToWordsIterative(int num)
        {
            // Handle the special case where the number is zero
            if (num == 0) return "Zero";

            // Arrays to store words for single digits, tens, and thousands
            String[] ones = { "", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen" };
            String[] tens = { "", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety" };
            String[] thousands = { "", "Thousand", "Million", "Billion" };

            // StringBuilder to accumulate the result
            StringBuilder result = new StringBuilder();
            int groupIndex = 0;

            // Process the number in chunks of 1000
            while (num > 0)
            {
                // Process the last three digits
                if (num % 1000 != 0)
                {
                    StringBuilder groupResult = new StringBuilder();
                    int part = num % 1000;

                    // Handle hundreds
                    if (part >= 100)
                    {
                        groupResult.Append(ones[part / 100]).Append(" Hundred ");
                        part %= 100;
                    }

                    // Handle tens and units
                    if (part >= 20)
                    {
                        groupResult.Append(tens[part / 10]).Append(" ");
                        part %= 10;
                    }

                    // Handle units
                    if (part > 0)
                    {
                        groupResult.Append(ones[part]).Append(" ");
                    }

                    // Append the scale (thousand, million, billion) for the current group
                    groupResult.Append(thousands[groupIndex]).Append(" ");
                    // Insert the group result at the beginning of the final result
                    result.Insert(0, groupResult);
                }
                // Move to the next chunk of 1000
                num /= 1000;
                groupIndex++;
            }

            return result.ToString().Trim();
        }
        public class NumberWord
        {
            public int Value;
            public string Word;

            public NumberWord(int value, String word)
            {
                this.Value = value;
                this.Word = word;
            }
        }

        private static readonly List<NumberWord> numberToWordsList = new List<NumberWord>
    {
        new NumberWord(1000000000, "Billion"),
        new NumberWord(1000000, "Million"),
        new NumberWord(1000, "Thousand"),
        new NumberWord(100, "Hundred"),
        new NumberWord(90, "Ninety"),
        new NumberWord(80, "Eighty"),
        new NumberWord(70, "Seventy"),
        new NumberWord(60, "Sixty"),
        new NumberWord(50, "Fifty"),
        new NumberWord(40, "Forty"),
        new NumberWord(30, "Thirty"),
        new NumberWord(20, "Twenty"),
        new NumberWord(19, "Nineteen"),
        new NumberWord(18, "Eighteen"),
        new NumberWord(17, "Seventeen"),
        new NumberWord(16, "Sixteen"),
        new NumberWord(15, "Fifteen"),
        new NumberWord(14, "Fourteen"),
        new NumberWord(13, "Thirteen"),
        new NumberWord(12, "Twelve"),
        new NumberWord(11, "Eleven"),
        new NumberWord(10, "Ten"),
        new NumberWord(9, "Nine"),
        new NumberWord(8, "Eight"),
        new NumberWord(7, "Seven"),
        new NumberWord(6, "Six"),
        new NumberWord(5, "Five"),
        new NumberWord(4, "Four"),
        new NumberWord(3, "Three"),
        new NumberWord(2, "Two"),
        new NumberWord(1, "One")
    };

        public string NumberToWordsPB(int num)
        {
            if (num == 0)
            {
                return "Zero";
            }

            foreach (NumberWord numberWord in numberToWordsList)
            {
                // Check if the number is greater than or equal to the current unit
                if (num >= numberWord.Value)
                {
                    // Convert the quotient to words if the current unit is 100 or greater
                    string prefix = (num >= 100) ? NumberToWords(num / numberWord.Value) + " " : "";

                    // Get the word for the current unit
                    string unit = numberWord.Word;

                    // Convert the remainder to words if it's not zero
                    string suffix = (num % numberWord.Value == 0) ? "" : " " + NumberToWords(num % numberWord.Value);

                    return prefix + unit + suffix;
                }
            }

            return "";
        }

        /*
        29. Divide Two Integers
        https://leetcode.com/problems/divide-two-integers/description/
        */
        public int Divide(int dividend, int divisor)
        {
            /*
Approach 1: Repeated Subtraction (RS)
Complexity Analysis
Let n be the absolute value of dividend.
•	Time Complexity : O(n).
Consider the worst case where the divisor is 1. For any dividend n, we'll need to subtract 1 a total of n times to get to 0. Therefore, the time complexity is O(n) in the worst case.
•	Space Complexity : O(1).
We only use a fixed number of integer variables, so the space complexity is O(1).
Seeing as n can be up to 231, this algorithm is too slow on the largest test cases. We'll need to do better!

            */
            int quotient = DivideRS(dividend, divisor);

            /*
Approach 2: Repeated Exponential Searches (RES)	            
Complexity Analysis
Let n be the absolute value of dividend.
•	Time Complexity : O(log2n).
We started by performing an exponential search to find the biggest number that fits into the current dividend. This search took O(logn) operations.
After doing this search, we updated the dividend by subtracting the number we found. In the worst case, we were left with a dividend slightly less than half of the previous dividend (if it was more than half, then we couldn't have found the maximum number that fit in by doubling!).
So how many of these searches did we need to do? Well, with the dividend at least halving after each one, there couldn't have been more than O(logn) of them.
So combined together, in the worst case, we have O(logn) searches with each search taking O(logn) time. This gives us O((logn)⋅(logn))=O(log2n) as our total time complexity.
•	Space Complexity : O(1).
Because only a constant number of single-value variables are used, the space complexity is O(1).


            */
            quotient = DivideRES(dividend, divisor);
            /*
  Approach 3: Adding Powers of Two (PT)          
  Complexity Analysis
Let n be the absolute value of dividend.
•	Time Complexity : O(logn).
We take O(logn) time in the first loop to create our list of doubles (and powers of two).
For the second loop, because there's O(logn) items in the list of doubles, it only takes O(logn)time for this loop as well.
Combined, our total time complexity is just O(logn+logn)=O(logn).
•	Space Complexity : O(logn).
The length of the list of doubles of the divisor is proportional to O(logn) so our space complexity is O(logn).
This approach is interesting in that the time complexity is lower than the previous one, but it requires a bit of space. Trading off space for time is very common practice.
However, as we'll see in the next approach, we can modify the algorithm so that we don't need O(logn) space at all!


            */
            quotient = DividePT(dividend, divisor);

            /*
  Approach 4: Adding Powers of Two with Bit-Shifting  (PTBS)
Complexity Analysis
Let n be the absolute value of dividend.
•	Time Complexity : O(logn).
Same as Approach 3, except instead of looping over a generated array, we simply perform an O(1) halving operation to get the next values we need.
•	Space Complexity : O(1).
We only use a fixed number of integer variables, so the space complexity is O(1).

            */
            quotient = DividePTBS(dividend, divisor);
            /*
  Approach 5: Binary Long Division (BLD)         
Complexity Analysis
Let n be the absolute value of dividend.
•	Time Complexity : O(logn).
As we loop over the bits of our dividend, performing an O(1) operation each time, the time complexity is just the number of bits of the dividend: O(logn).
•	Space Complexity : O(1).
We only use a fixed number of int variables, so the space complexity is O(1).

            */

            quotient = DivideBLD(dividend, divisor);

            return quotient;

        }
        public int DivideRS(int dividend, int divisor)
        {
            // Special case: overflow.
            if (dividend == int.MinValue && divisor == -1)
            {
                return int.MaxValue;
            }

            /* We need to convert both numbers to negatives
             * for the reasons explained above.
             * Also, we count the number of negatives signs. */
            int negatives = 2;
            if (dividend > 0)
            {
                negatives--;
                dividend = -dividend;
            }

            if (divisor > 0)
            {
                negatives--;
                divisor = -divisor;
            }

            /* Count how many times the divisor has to be added
             * to get the dividend. This is the quotient. */
            int quotient = 0;
            while (dividend - divisor <= 0)
            {
                quotient--;
                dividend -= divisor;
            }

            /* If there was originally one negative sign, then
             * the quotient remains negative. Otherwise, switch
             * it to positive. */
            if (negatives != 1)
            {
                quotient = -quotient;
            }

            return quotient;
        }
        int HALF_INT_MIN = -1073741824;

        public int DivideRES(int dividend, int divisor)
        {
            // Special case: overflow.
            if (dividend == Int32.MinValue && divisor == -1)
            {
                return Int32.MaxValue;
            }

            /* We need to convert both numbers to negatives.
             * Also, we count the number of negatives signs. */
            int negatives = 2;
            if (dividend > 0)
            {
                negatives--;
                dividend = -dividend;
            }

            if (divisor > 0)
            {
                negatives--;
                divisor = -divisor;
            }

            int quotient = 0;
            /* Once the divisor is bigger than the current dividend,
             * we can't fit any more copies of the divisor into it. */
            while (divisor >= dividend)
            {
                /* We know it'll fit at least once as divivend >= divisor.
                 * Note: We use a negative powerOfTwo as it's possible we might have
                 * the case divide(Int32.MinValue, -1). */
                int powerOfTwo = -1;
                int value = divisor;
                /* Check if double the current value is too big. If not, continue
                 * doubling. If it is too big, stop doubling and continue with the
                 * next step */
                while (value >= HALF_INT_MIN && value + value >= dividend)
                {
                    value += value;
                    powerOfTwo += powerOfTwo;
                }

                // We have been able to subtract divisor another powerOfTwo times.
                quotient += powerOfTwo;
                // Remove value so far so that we can continue the process with
                // remainder.
                dividend -= value;
            }

            /* If there was originally one negative sign, then
             * the quotient remains negative. Otherwise, switch
             * it to positive. */
            return negatives != 1 ? -quotient : quotient;
        }
        public int DividePT(int dividend, int divisor)
        {
            // Special case: overflow.
            if (dividend == int.MinValue && divisor == -1)
            {
                return int.MaxValue;
            }

            /* We need to convert both numbers to negatives.
             * Also, we count the number of negatives signs. */
            int negatives = 2;
            if (dividend > 0)
            {
                negatives--;
                dividend = -dividend;
            }

            if (divisor > 0)
            {
                negatives--;
                divisor = -divisor;
            }

            List<int> doubles = new List<int>();
            List<int> powersOfTwo = new List<int>();
            /* Nothing too exciting here, we're just making a list of doubles of 1
             * and the divisor. This is pretty much the same as Approach 2, except
             * we're actually storing the values this time. */
            int powerOfTwo = -1;
            while (divisor >= dividend)
            {
                doubles.Add(divisor);
                powersOfTwo.Add(powerOfTwo);
                // Prevent needless overflows from occurring...
                if (divisor < HALF_INT_MIN)
                {
                    break;
                }

                divisor += divisor;
                powerOfTwo += powerOfTwo;
            }

            int quotient = 0;
            /* Go from largest double to smallest, checking if the current double
             * fits. into the remainder of the dividend. */
            for (int i = doubles.Count - 1; i >= 0; i--)
            {
                if (doubles[i] >= dividend)
                {
                    // If it does fit, add the current powerOfTwo to the quotient.
                    quotient += powersOfTwo[i];
                    // Update dividend to take into account the bit we've now
                    // removed.
                    dividend -= doubles[i];
                }
            }

            /* If there was originally one negative sign, then
             * the quotient remains negative. Otherwise, switch
             * it to positive. */
            if (negatives != 1)
            {
                return -quotient;
            }

            return quotient;
        }
        public int DividePTBS(int dividend, int divisor)
        {
            const int INT_MIN = -2147483648;
            const int INT_MAX = 2147483647;
            const int HALF_INT_MIN = -1073741824;
            // Special case: overflow.
            if (dividend == INT_MIN && divisor == -1)
            {
                return INT_MAX;
            }

            int negatives = 2;
            if (dividend > 0)
            {
                negatives--;
                dividend = -dividend;
            }

            if (divisor > 0)
            {
                negatives--;
                divisor = -divisor;
            }

            int highestDouble = divisor;
            int highestPowerOfTwo = -1;
            while (highestDouble >= HALF_INT_MIN &&
                   dividend <= highestDouble + highestDouble)
            {
                highestPowerOfTwo += highestPowerOfTwo;
                highestDouble += highestDouble;
            }

            int quotient = 0;
            while (dividend <= divisor)
            {
                if (dividend <= highestDouble)
                {
                    quotient += highestPowerOfTwo;
                    dividend -= highestDouble;
                }

                highestPowerOfTwo >>= 1;
                highestDouble >>= 1;
            }

            if (negatives != 1)
            {
                return -quotient;
            }

            return quotient;
        }


        public int DivideBLD(int dividend, int divisor)
        {
            // Special cases: overflow.
            if (dividend == int.MinValue && divisor == -1)
            {
                return int.MaxValue;
            }

            if (dividend == int.MinValue && divisor == 1)
            {
                return int.MinValue;
            }

            /* We need to convert both numbers to negatives.
             * Also, we count the number of negatives signs. */
            int negatives = 2;
            if (dividend > 0)
            {
                negatives--;
                dividend = -dividend;
            }

            if (divisor > 0)
            {
                negatives--;
                divisor = -divisor;
            }

            /* We want to find the largest doubling of the divisor in the negative
             * 32-bit integer range that could fit into the dividend. Note if it
             * would cause an overflow by being less than HALF_INT_MIN, then we just
             * stop as we know double it would not fit into INT_MIN anyway. */
            int maxBit = 0;
            while (divisor >= HALF_INT_MIN && divisor + divisor >= dividend)
            {
                maxBit += 1;
                divisor += divisor;
            }

            int quotient = 0;
            /* We start from the biggest bit and shift our divisor to the right
             * until we can't shift it any further */
            for (int bit = maxBit; bit >= 0; bit--)
            {
                /* If the divisor fits into the dividend, then we should set the
                 * current bit to 1. We can do this by subtracting a 1 shifted by
                 * the appropriate number of bits. */
                if (divisor >= dividend)
                {
                    quotient -= (1 << bit);
                    /* Remove the current divisor from the dividend, as we've now
                     * considered this part. */
                    dividend -= divisor;
                }

                /* Shift the divisor to the right so that it's in the right place
                 * for the next position we're checking at. */
                divisor = (divisor + 1) >> 1;
            }

            /* If there was originally one negative sign, then
             * the quotient remains negative. Otherwise, switch
             * it to positive. */
            if (negatives != 1)
            {
                quotient = -quotient;
            }

            return quotient;
        }

        /*
        38. Count and Say
        https://leetcode.com/problems/count-and-say/description/

        */
        public string CountAndSay(int n)
        {
            /*
            
Approach 1: Straightforward
Complexity Analysis
•	Time Complexity: O(4^(n/3)).
•	Space Complexity: O(4^(n/3)).

            */
            string result = CountAndSay(n);

            /*
            Approach 2: Regular Expression
        Complexity Analysis
        •	Time Complexity: O(4^(n/3))
         •	Space Complexity: O(4^(n/3))   
            */

            result = CountAndSayRegEx(n);

            return result;
        }

        public string CountAndSay1(int n)
        {
            string currentString = "1";
            for (int i = 2; i <= n; i++)
            {
                string nextString = "";
                for (int j = 0, k = 0; j < currentString.Length; j = k)
                {
                    while (k < currentString.Length &&
                           currentString[k] == currentString[j])
                        k++;
                    nextString += (k - j).ToString() + currentString[j];
                }

                currentString = nextString;
            }

            return currentString;
        }
        public string CountAndSayRegEx(int n)
        {
            string currentString = "1";
            // pattern to match the repetitive digits
            Regex pattern = new Regex(@"(.)(\1)*");

            for (int i = 1; i < n; ++i)
            {
                MatchCollection matches = pattern.Matches(currentString);
                StringBuilder nextString = new StringBuilder();

                // each group contains identical and adjacent digits
                foreach (Match m in matches)
                {
                    nextString.Append(m.Length.ToString() + m.Value[0]);
                }

                // prepare for the next iteration
                currentString = nextString.ToString();
            }

            return currentString;
        }
    }
}