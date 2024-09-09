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
        public class NthFibSol
        {

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
        }


        /*
    600. Non-negative Integers without Consecutive Ones
    https://leetcode.com/problems/non-negative-integers-without-consecutive-ones/description/

            */
        public class NonNegIntWOConsOnesSol
        {

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

        }

        /*
        12. Integer to Roman
https://leetcode.com/problems/integer-to-roman/description/	
     
        */
        public class IntToRomanSol
        {
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
        }
        /*
273. Integer to English Words
https://leetcode.com/problems/integer-to-english-words/	

        */
        public class NumberToWordsSol
        {
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
        }
        /*
        29. Divide Two Integers
        https://leetcode.com/problems/divide-two-integers/description/
        */
        public class DivideTwoIntSol
        {
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
        }
        /*
        38. Count and Say
        https://leetcode.com/problems/count-and-say/description/

        */
        public class CountAndSaySol
        {
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
        /*
        43. Multiply Strings
        https://leetcode.com/problems/multiply-strings/description/

        */
        public class MultiplStringsSol
        {
            public string MultiplyStrings(string num1, string num2)
            {
                /*
    Approach 1: Elementary Math (EM)
    Here N and M are the number of digits in num1 and num2 respectively.
    •	Time complexity: O(M^2+M⋅N).
    •	Space complexity: O(M^2+M⋅N).
                */
                string results = MultiplyStringsEM(num1, num2);
                /*
    Approach 2: Elementary math using less intermediate space (EMS)
     Complexity Analysis
    Here N and M are the number of elements in num 1 and num 2 strings.
    •	Time complexity: O(M⋅(N+M)).
    •	Space complexity: O(N+M).
    o	The answer string and multiplication results will have at most N+M length.

                */
                results = MultiplyStringsEMS(num1, num2);
                /*
    Approach 3: Sum the products from all pairs of digits (SPPD)          
    Complexity Analysis
    Here N and M are the number of digits in num1 and num2 respectively.
    •	Time complexity: O(M⋅N).
    •	Space complexity: O(M+N).
                */
                results = MultiplyStringsSPPD(num1, num2);

                return results;

            }
            // Calculate the sum of all of the results from multiplyOneDigit.
            private List<int> SumResults(List<List<int>> results)
            {
                // Initialize answer as a number from results.
                List<int> answer = new List<int>(results[results.Count - 1]);
                List<int> newAnswer;
                for (int j = 0; j < results.Count - 1; ++j)
                {
                    List<int> result = new List<int>(results[j]);
                    newAnswer = new List<int>();
                    int carry = 0;
                    for (int i = 0; i < answer.Count || i < result.Count; ++i)
                    {
                        int digit1 = i < result.Count ? result[i] : 0;
                        int digit2 = i < answer.Count ? answer[i] : 0;
                        int sum = digit1 + digit2 + carry;
                        carry = sum / 10;
                        newAnswer.Add(sum % 10);
                    }

                    if (carry != 0)
                    {
                        newAnswer.Add(carry);
                    }

                    answer = newAnswer;
                }

                return answer;
            }

            private List<int> MultiplyOneDigit(char[] firstNumber,
                                               char secondNumberDigit, int numZeros)
            {
                List<int> currentResult = Enumerable.Repeat(0, numZeros).ToList();
                int carry = 0;
                for (int i = 0; i < firstNumber.Length; i++)
                {
                    int multiplication =
                        (secondNumberDigit - '0') * (firstNumber[i] - '0') + carry;
                    carry = multiplication / 10;
                    currentResult.Add(multiplication % 10);
                }

                if (carry != 0)
                {
                    currentResult.Add(carry);
                }

                return currentResult;
            }

            public string MultiplyStringsEM(string num1, string num2)
            {
                if (num1 == "0" || num2 == "0")
                {
                    return "0";
                }

                char[] firstNumber = num1.ToCharArray();
                Array.Reverse(firstNumber);
                char[] secondNumber = num2.ToCharArray();
                Array.Reverse(secondNumber);
                List<List<int>> results = new List<List<int>>();
                for (int i = 0; i < secondNumber.Length; ++i)
                {
                    results.Add(MultiplyOneDigit(firstNumber, secondNumber[i], i));
                }

                List<int> answer = SumResults(results);
                return string.Join(
                    "", answer.Select(t => t.ToString()).ToArray().Reverse());
            }
            // Function to add two strings.
            private List<int> AddStrings(List<int> num1, List<int> num2)
            {
                List<int> ans = new List<int>();
                int carry = 0;

                for (int i = 0; i < num1.Count || i < num2.Count || carry != 0; ++i)
                {
                    // If num2 is shorter than num1 or vice versa, use 0 as the current
                    // digit.
                    int digit1 = i < num1.Count ? num1[i] : 0;
                    int digit2 = i < num2.Count ? num2[i] : 0;

                    // Add current digits of both numbers.
                    int sum = digit1 + digit2 + carry;
                    // Set carry equal to the tens place digit of sum.
                    carry = sum / 10;
                    // Append the ones place digit of sum to answer.
                    ans.Add(sum % 10);
                }

                if (carry != 0)
                {
                    ans.Add(carry);
                }

                return ans;
            }

            // Multiply the current digit of secondNumber with firstNumber.
            private List<int> MultiplyOneDigit(StringBuilder firstNumber,
                                               char secondNumberDigit, int numZeros)
            {
                // Insert zeros at the beginning based on the current digit's place.
                List<int> currentResult = new List<int>(new int[numZeros]);

                int carry = 0;

                // Multiply firstNumber with the current digit of secondNumber.
                for (int i = 0; i < firstNumber.Length; ++i)
                {
                    int multiplication =
                        (secondNumberDigit - '0') * (firstNumber[i] - '0') + carry;
                    // Set carry equal to the tens place digit of multiplication.
                    carry = multiplication / 10;
                    // Append last digit to the current result.
                    currentResult.Add(multiplication % 10);
                }

                if (carry != 0)
                {
                    currentResult.Add(carry);
                }

                return currentResult;
            }

            public string MultiplyStringsEMS(string num1, string num2)
            {
                if (num1 == "0" || num2 == "0")
                {
                    return "0";
                }

                // Reverse both the numbers.
                StringBuilder firstNumber =
                    new StringBuilder(new string(num1.Reverse().ToArray()));
                StringBuilder secondNumber =
                    new StringBuilder(new string(num2.Reverse().ToArray()));

                // To store the multiplication result of each digit of secondNumber with
                // firstNumber.
                int N = firstNumber.Length + secondNumber.Length;
                List<int> ans = new List<int>(new int[N]);

                // For each digit in secondNumber, multiply the digit by firstNumber and
                // add the multiplication result to ans.
                for (int i = 0; i < secondNumber.Length; ++i)
                {
                    // Add the current result to final ans.
                    ans = AddStrings(MultiplyOneDigit(firstNumber, secondNumber[i], i),
                                     ans);
                }

                // Pop excess 0 from the rear of ans.
                if (ans[ans.Count - 1] == 0)
                {
                    ans.RemoveAt(ans.Count - 1);
                }

                // Ans is in the reversed order.
                // Copy it in reverse order to get the final ans.
                StringBuilder answer = new StringBuilder();
                for (int i = ans.Count - 1; i >= 0; --i)
                {
                    answer.Append(ans[i]);
                }

                return answer.ToString();
            }

            public string MultiplyStringsSPPD(string num1, string num2)
            {
                if (num1.Equals("0") || num2.Equals("0"))
                {
                    return "0";
                }

                char[] firstNumber = num1.ToCharArray();
                char[] secondNumber = num2.ToCharArray();
                Array.Reverse(firstNumber);
                Array.Reverse(secondNumber);
                // To store the multiplication result of each digit of secondNumber with
                // firstNumber.
                int firstNumLength = firstNumber.Length;
                int secondNumLength = secondNumber.Length;
                int resultArrayLength = firstNumLength + secondNumLength;
                int[] resultArray = new int[resultArrayLength];
                for (int place2 = 0; place2 < secondNumLength; place2++)
                {
                    int digit2 = secondNumber[place2] - '0';
                    // For each digit in secondNumber multiply the digit by all digits
                    // in firstNumber.
                    for (int place1 = 0; place1 < firstNumLength; place1++)
                    {
                        int digit1 = firstNumber[place1] - '0';
                        // The number of zeros from multiplying to digits depends on the
                        // place of digit2 in secondNumber and the place of the digit1
                        // in firstNumber.
                        int numZeros = place1 + place2;
                        int multiplication = digit1 * digit2 + resultArray[numZeros];
                        // Set the ones place of the multiplication result.
                        resultArray[numZeros] = multiplication % 10;
                        // Carry the tens place of the multiplication result by
                        // adding it to the next position in the answer array.
                        resultArray[numZeros + 1] += multiplication / 10;
                    }
                }

                // Pop excess 0s from the rear of answer.
                if (resultArray[resultArray.Length - 1] == 0)
                {
                    resultArray = resultArray.Take(resultArray.Length - 1).ToArray();
                }

                // Ans is in the reversed order.
                // Reverse it to get the final ans.
                Array.Reverse(resultArray);
                return string.Join(
                    string.Empty,
                    resultArray.Select(digit => digit.ToString()).ToArray());
            }
        }

        /*
        50. Pow(x, n)
        https://leetcode.com/problems/powx-n/description/

        */
        public class MyPowSol
        {
            public double MyPow(double x, int n)
            {
                /*

    Approach 1: Binary Exponentiation (Recursive) (BEI)
     Complexity Analysis
    •	Time complexity: O(logn)
    o	At each recursive call we reduce n by half, so we will make only logn number of calls for the binaryExp function, and the multiplication of two numbers is considered as a constant time operation.
    o	Thus, it will take overall O(logn) time.
     •	Space complexity: O(logn)
    o	The recursive stack can use at most O(logn) space at any time.        


                */

                double pow1 = MyPowBER(x, n);

                /*
     Approach 2: Binary Exponentiation (Iterative) (BEI)           
       Complexity Analysis
    •	Time complexity: O(logn)
    o	At each iteration, we reduce n by half, thus it means we will make only logn number of iterations using a while loop.
    o	Thus, it will take overall O(logn) time.
    •	Space complexity: O(1)
    o	We don't use any additional space.


                */
                pow1 = MyPowBEI(x, n);

                return pow1;

            }

            private double BinaryExp(double x, long n)
            {
                // Base case, to stop recursive calls.
                if (n == 0)
                {
                    return 1;
                }

                // Handle case where, n < 0.
                if (n < 0)
                {
                    return 1.0 / BinaryExp(x, -1 * n);
                }

                // Perform Binary Exponentiation.
                // If 'n' is odd we perform Binary Exponentiation on 'n - 1' and
                // multiply result with 'x'.
                if (n % 2 == 1)
                {
                    return x * BinaryExp(x * x, (n - 1) / 2);
                }
                // Otherwise we calculate result by performing Binary Exponentiation on
                // 'n'.
                else
                {
                    return BinaryExp(x * x, n / 2);
                }
            }

            public double MyPowBER(double x, int n)
            {
                // Call recursive function with correct types.
                return BinaryExp(x, (long)n);
            }

            private double BinaryExpBEI(double x, long n)
            {
                if (n == 0)
                {
                    return 1;
                }

                // Handle case where, n < 0.
                if (n < 0)
                {
                    n = -1 * n;
                    x = 1.0 / x;
                }

                // Perform Binary Exponentiation.
                double result = 1;
                while (n != 0)
                {
                    // If 'n' is odd we multiply result with 'x' and reduce 'n' by '1'.
                    if (n % 2 == 1)
                    {
                        result *= x;
                        n -= 1;
                    }

                    // We square 'x' and reduce 'n' by half, x^n => (x^2)^(n/2).
                    x *= x;
                    n /= 2;
                }

                return result;
            }

            public double MyPowBEI(double x, int n)
            {
                return BinaryExpBEI(x, (long)n);
            }
        }


        /*
        69. Sqrt(x)
        https://leetcode.com/problems/sqrtx/description/
        */
        public class SqrtSol
        {
            /*
            Approach 1: Pocket Calculator Algorithm (PCA)
            Complexity Analysis
    •	Time complexity: O(1).
    •	Space complexity: O(1).

            */
            public int MySqrtPCA(int x)
            {
                if (x < 2)
                    return x;
                long left = (long)Math.Exp(0.5 * Math.Log(x));
                long right = left + 1;
                return right * right > x ? (int)left : (int)right;
            }

            /*
            Approach 2: Binary Search
    Complexity Analysis
    •	Time complexity : O(logN).
    Let's compute time complexity with the help of master theorem T(N)=aT(bN)+Θ(Nd). The equation represents dividing the problem up into a subproblems of size bN in Θ(Nd) time. Here at step, there is only one subproblem a = 1, its size is half of the initial problem b = 2, and all this happens in a constant time d = 0. That means that logba=d and hence we're dealing with case 2 that results in O(nlogbalogd+1N) = O(logN) time complexity.
    •	Space complexity : O(1).


            */
            public int MySqrtBS(int x)
            {
                if (x < 2)
                    return x;
                long num;
                int pivot, left = 2, right = x / 2;
                while (left <= right)
                {
                    pivot = left + (right - left) / 2;
                    num = (long)pivot * pivot;
                    if (num > x)
                        right = pivot - 1;
                    else if (num < x)
                        left = pivot + 1;
                    else
                        return pivot;
                }

                return right;
            }

            /*
            Approach 3: Recursion + Bit Shifts (RBS)
            Complexity Analysis
•	Time complexity: O(logN).
•	Space complexity: O(logN) to keep the recursion stack.
            */
            public int MySqrtRBS(int x)
            {
                if (x < 2)
                    return x;
                int left = MySqrtRBS(x >> 2) << 1;
                int right = left + 1;
                return (long)right * right > x ? left : right;
            }
            /*
            Approach 4: Newton's Method (NM)
    Complexity Analysis
    •	Time complexity: O(logN).
    •	Space complexity: O(1).
            */
            public int MySqrtNM(int x)
            {
                if (x < 2)
                    return x;
                double x0 = x;
                double x1 = (x0 + x / x0) / 2.0;
                while (Math.Abs(x0 - x1) >= 1)
                {
                    x0 = x1;
                    x1 = (x0 + x / x0) / 2.0;
                }

                return (int)x1;
            }


        }


        /*
        89. Gray Code
        https://leetcode.com/problems/gray-code/description/
        */
        public class GrayCodeSol
        {
            /*
            Approach 1: Backtracking
          Complexity Analysis
Here n is the total number of bits in the Gray code.
•	Time complexity: O(n∗2^n)
For n bits 2^n numbers are possible. The maximum depth of the recursive function stack is 2n (The recursion stops when the size of result list is 2^n).
In our backtracking algorithm, at every function call we iterate over a loop of length n and try to find out all possible successors of the last added number in the present sequence. We continue our search with the first value that succeeds (not present in the isPresent set). Since we are using HashSet and unordered_set which have an amortized runtime of O(1) in all operations, use of sets doesn't increase the time complexity.
 •	Space complexity: O(2^n)
We use a set isPresent which will contain at most 2n numbers. The space occupied by the output result is not considered in the space complexity analysis.
 
            */
            private List<int> result;
            private HashSet<int> isPresent;

            public List<int> GrayCodeBackTrack(int n)
            {
                result = new List<int> { 0 };
                // Keeps track of the numbers present in the current sequence.
                // All Gray code sequence starts with 0
                isPresent = new HashSet<int> { 0 };

                // Create a new thread with increased stack size
                Thread thread = new Thread(() => GrayCodeHelper(0, n),
                                           1024 * 1024 * 10);  // 10 MB stack
                thread.Start();
                thread.Join();  // Wait for the thread to complete

                return result;
            }

            private bool GrayCodeHelper(int current, int n)
            {
                if (result.Count == (1 << n))
                    return true;

                for (int i = 0; i < n; i++)
                {
                    int next = current ^ (1 << i);
                    if (!isPresent.Contains(next))
                    {
                        isPresent.Add(next);
                        result.Add(next);
                        if (GrayCodeHelper(next, n))
                            return true;  // Early exit on success

                        // If no valid sequence found, backtrack
                        isPresent.Remove(next);
                        result.RemoveAt(result.Count - 1);
                    }
                }

                return false;
            }
            /*
            Approach 2: Recursion
Complexity Analysis
Here n is the total number of bits in the code.
•	Time complexity: O(2^n)
The maximum depth of the recursive function stack is n. At every function call we iterate over the list result and at each iteration we add a new number to the sequence. At n=0 the size of the list is 1. At n=1 we iterate over the list [0] of size 20=1. Next, at n=2 we iterate over the list [0,1] of size 21=2. Likewise, at n=3 we iterate over the list [0,1,3,2] of size 22=4. This ends at n where we iterate over a list of size 2^n−1.
The mathematical expression for the above analysis reflects a geometric progression as follows 1+2+4+8+......2n−1. Thus the time complexity will be O(2^n).
•	Space complexity: O(n)
We start from n and continue our recursive function call until our base condition n=0 is reached. Thus, the depth of the function call stack will be O(n).


            */


            public IList<int> GrayCodeRec(int n)
            {
                result = new List<int>();
                GrayCodeHelper(n);
                return result;
            }

            void GrayCodeHelper(int n)
            {
                if (n == 0)
                {
                    result.Add(0);
                    return;
                }

                // derive the n bits sequence from the (n - 1) bits sequence.
                GrayCodeHelper(n - 1);
                int currentSequenceLength = result.Count;
                // Set the bit at position n - 1 (0 indexed) and assign it to mask.
                int mask = 1 << (n - 1);
                for (int i = currentSequenceLength - 1; i >= 0; i--)
                {
                    // mask is used to set the (n - 1)th bit from the LSB of all the
                    // numbers present in the current sequence.
                    result.Add(result[i] | mask);
                }

                return;
            }


            /*
            Approach 3: Iteration

           Complexity Analysis
        Here n is the total number of bits in the Gray code.
        •	Time complexity: O(2^n)
        One insight regarding the time complexity is that in the iterative algorithm, we are building the list in a deterministic and non-redundant way. This means that there is no backtracking, each iteration will produce a series of valid elements in the final sequence.
        Since the total number of elements in the final sequence is 2^n, it will take a total of 2^n steps to produce the sequence.
        Here the outer loop runs from i=1ton. At ith step the length of the result list is 2i−1. So the number of iterations at n=1 is 1, at n=2 is 2, n=3 is 4 and so on. Similar to Approach 2 the mathematical expression for the above analysis reflects a geometric progression as follows 1+2+4+8+...+2^n−1.
         Thus the time complexity will be O(2^n).
        •	Space complexity: O(1)
        The space occupied by the output result is not considered in the space complexity analysis. So overall no extra space is required in this approach.


            */
            public IList<int> GrayCodeIteration(int n)
            {
                IList<int> result = new List<int>();
                result.Add(0);
                for (int i = 1; i <= n; i++)
                {
                    int previousSequenceLength = result.Count;
                    int mask = 1 << (i - 1);
                    for (int j = previousSequenceLength - 1; j >= 0; j--)
                    {
                        result.Add(mask + result[j]);
                    }
                }

                return result;
            }


            /*
            Approach 4: Recursion 2
Complexity Analysis
Here n is the total number of bits in the code.
•	Time complexity: O(2^n)
The above diagram illustrates the recursion tree. If we consider each node as a block of statements executed at each recursive function call, the total time complexity will be in terms of the number of nodes present. For n bits the total number of nodes (internal + leaf) is 2^(n+1)−1. So the overall time complexity is O(2^(n+1)−1) = O(2^n).
•	Space complexity: O(n)
From the diagram, we can conclude that the maximum depth of the function call stack will be n+1. So the space complexity will be O(n). The space occupied by the result is not considered in the space complexity analysis.


    */
            private int nextNum = 0;

            public IList<int> GrayCodeRec2(int n)
            {
                List<int> result = new List<int>();
                GrayCodeHelper(result, n);
                return result;
            }

            private void GrayCodeHelper(List<int> result, int n)
            {
                if (n == 0)
                {
                    result.Add(this.nextNum);
                    return;
                }

                GrayCodeHelper(result, n - 1);
                // Flip the bit at (n - 1)th position from right
                this.nextNum = this.nextNum ^ (1 << (n - 1));
                GrayCodeHelper(result, n - 1);
            }
            /*
            Approach 5: Iteration with a single loop (ISL)
Complexity Analysis
Here n is the total number of bits in the Gray code.
•	Time complexity: O(2^n)
We use a single for loop to add all the 2^n numbers of the Gray code sequence to the result. Thus the time complexity will be O(2^n).
•	Space complexity: O(1)
The space occupied by the output result is not considered in the space complexity analysis. So overall no extra space is used in this approach.


            */
            public IList<int> GrayCodeISL(int n)
            {
                IList<int> result = new List<int>();
                // there are 2 ^ n numbers in the Gray code sequence.
                int sequenceLength = 1 << n;
                for (int i = 0; i < sequenceLength; i++)
                {
                    int num = i ^ i >> 1;
                    result.Add(num);
                }

                return result;
            }


        }





















    }
}