using System;
using System.Buffers;
using System.Collections.Generic;
using System.Dynamic;
using System.Globalization;
using System.Linq;
using System.Reflection;
using System.Reflection.Metadata.Ecma335;
using System.Reflection.PortableExecutable;
using System.Runtime.Versioning;
using System.Text;
using System.Threading.Tasks;
using AlgoDSPlay.Combinatoric.Enumeration;
using AlgoDSPlay.DataStructures;
using AlgoDSPlay.Design;

namespace AlgoDSPlay
{
    public class ArrayOps
    {
        //https://www.algoexpert.io/questions/two-number-sum
        public static int[] TwoNumberSum(int[] array, int targetSum)
        {


            //1.Naive - T:O(n^2) | O(1) - Pair of loops
            var result = TwoNumberSumNaive(array, targetSum);
            //2.Optimal - T: O(n) | O(n) - 
            result = TwoNumberSumOptimal(array, targetSum);

            //3.Optimal with Sorting - T: O(nlog(n)) | O(1)
            result = TwoNumberSumOptimal2(array, targetSum);
            return result;

        }

        private static int[] TwoNumberSumOptimal2(int[] array, int targetSum)
        {
            //IntroSort: insertion sort, heapsort and quick sort based on partitions and call stack            
            Array.Sort(array);
            int left = 0, right = array.Length - 1;

            while (left < right)
            {

                int currentSum = array[left] + array[right];

                if (currentSum == targetSum)
                {
                    return new int[] { array[left], array[right] };
                }
                else if (currentSum < targetSum)
                {
                    left++;
                }
                else if (currentSum > targetSum)
                    right--;
            }
            return new int[0];

        }
        // O(n^2) time | O(1) space
        public static int[] TwoNumberSumNaive(int[] array, int targetSum)
        {
            for (int i = 0; i < array.Length - 1; i++)
            {
                int firstNum = array[i];
                for (int j = i + 1; j < array.Length; j++)
                {
                    int secondNum = array[j];
                    if (firstNum + secondNum == targetSum)
                    {
                        return new int[] { firstNum, secondNum };
                    }
                }
            }
            return new int[0];
        }
        //https://www.algoexpert.io/questions/max-profit-with-k-transactions
        public static int MaxProfitWithKTransactions(int[] prices, int k)
        {
        //1. T:O(n^2k) | O(nk)
        TODO:
            //2. T:O(nk) | O(k)
            if (prices.Length == 0) return 0;

            int[,] profits = new int[k + 1, prices.Length];

            for (int t = 1; t < k + 1; t++)
            {
                int maxThusFar = Int32.MinValue;
                for (int d = 1; d < prices.Length; d++)
                {
                    maxThusFar = Math.Max(maxThusFar, profits[t - 1, d - 1] - prices[d - 1]);
                    profits[t, d] = Math.Max(profits[t, d - 1], maxThusFar + prices[d]);
                }

            }
            //3: T:O(nk) | S:O(n)
            // TODO:

            return profits[k, prices.Length - 1];

        }

        private static int[] TwoNumberSumOptimal(int[] array, int targetSum)
        {
            HashSet<int> set = new HashSet<int>();

            foreach (int val in array)
            {
                int potentialMatch = targetSum - val;
                if (set.Contains(potentialMatch))
                    return new int[] { val, potentialMatch };
                else
                    set.Add(val);
            }
            return new int[0];


        }
        //https://www.algoexpert.io/questions/single-cycle-check
        public static bool HasSingleCycle(int[] array)
        {
            //T:O(n)| S:O(1)
            int numElementsVisited = 0;
            int currentIdx = 0; //Start Idx

            while (numElementsVisited < array.Length)
            {

                if (numElementsVisited > 0 && currentIdx == 0) return false; //More than one cycle exists as in [1,2,1,-3,6,7]
                numElementsVisited++;
                currentIdx = GetNextIndex(currentIdx, array);
            }
            return currentIdx == 0;

        }

        private static int GetNextIndex(int currentIdx, int[] array)
        {
            int jump = array[currentIdx];
            int nextIdx = (currentIdx + jump) % array.Length; // in a scenario of index out of bound, [26, 1 3,5]
            nextIdx = nextIdx > 0 ? nextIdx : nextIdx + array.Length; //In case, nextIdx is negative        
            return nextIdx;
        }
        //https://www.algoexpert.io/questions/index-equals-value
        public static int IndexEqualsValue(int[] array)
        {
            //1.Naive - T:O(n) | S:O(1)
            int firstIndexEaqualsValue = IndexEqualsValueNaive(array);

            //2.Optimal using Binary Search as array is sorted
            //T:O(log(n)) | S:O(1)
            firstIndexEaqualsValue = IndexEqualsValueOptimal(array);

            //3.Optimal Recursive using Binary Search as array is sorted
            //T:O(log(n)) | S:O(1)
            firstIndexEaqualsValue = IndexEqualsValueOptimalRecursive(array, 0, array.Length - 1);

            return firstIndexEaqualsValue;
        }

        private static int IndexEqualsValueOptimalRecursive(int[] array, int leftIndex, int rightIndex)
        {
            if (leftIndex > rightIndex) return -1;

            int midIndex = leftIndex + (leftIndex - rightIndex) / 2;
            int midValue = array[midIndex];
            if (midValue < midIndex)
            {
                return IndexEqualsValueOptimalRecursive(array, midIndex + 1, rightIndex);
            }
            else if (midIndex == midValue && midIndex == 0)
                return midIndex;
            else if (midIndex == midValue && array[midIndex - 1] < midIndex - 1)
                return midIndex;
            else
                return IndexEqualsValueOptimalRecursive(array, leftIndex, midIndex - 1);
        }

        private static int IndexEqualsValueOptimal(int[] array)
        {
            int start = 0;
            int end = array.Length - 1;

            while (start <= end)
            {

                int midIndex = (start + end) / 2; // OR end+(start-end)/2

                int midValue = array[midIndex];

                if (midValue < midIndex)
                {
                    start = midIndex + 1;
                }
                else if (midIndex == midValue && midIndex == 0)
                {
                    return midIndex;
                }
                else if (midValue == midIndex && array[midIndex - 1] < midIndex - 1)
                {
                    return midIndex;
                }
                else { end = midIndex - 1; }


            }
            return -1;
        }

        private static int IndexEqualsValueNaive(int[] array)
        {
            for (int index = 0; index < array.Length; index++)
            {
                int value = array[index];
                if (index == value)
                    return index;
            }
            return -1;
        }
        //https://www.algoexpert.io/questions/permutations
        public static List<List<int>> GetPermutations(List<int> array)
        {
            List<List<int>> permutations = new List<List<int>>();
            //1. Upper Bound => T:O(n^2*n!) |S:O(n*n!)
            //Roughly=> T:O(n*n!) | S:O(n*n!)
            GetPermutations(array, new List<int>(), permutations);

            //2.T:O(n*n!) |S:O(n*n!) - without using immutable Lists
            GetPermutations(0, array, permutations);

            return permutations;
        }

        private static void GetPermutations(int i, List<int> array, List<List<int>> permutations)
        {
            if (i == array.Count - 1)
            {
                permutations.Add(new List<int>(array)); //O(n) for array creation
            }
            else
            {
                for (int j = i; j < array.Count; j++)
                { // O(n)
                    Swap(array, i, j); // O(1)
                    GetPermutations(i + 1, array, permutations); ////O(n!) calls 
                    Swap(array, i, j); // O(1)
                }
            }

        }

        private static void Swap(List<int> array, int i, int j)
        {
            int tmp = array[i];
            array[j] = array[i];
            array[i] = tmp;
        }
        private static void Swap(int[] array, int i, int j)
        {
            int tmp = array[i];
            array[j] = array[i];
            array[i] = tmp;
        }
        private static void GetPermutations(List<int> array, List<int> currentPerm, List<List<int>> permutations)
        {
            if (array.Count == 0 && currentPerm.Count > 0)
            {
                permutations.Add(currentPerm); //O(n!)
            }
            else
            {
                for (int i = 0; i < array.Count; i++)
                { // O(n)
                    List<int> newArray = new List<int>(array); // O(n)
                    newArray.RemoveAt(i);
                    List<int> newPerm = new List<int>(currentPerm); // O(n)
                    newPerm.Add(array[i]);
                    GetPermutations(newArray, newPerm, permutations);

                }
            }

        }
        //https://www.algoexpert.io/questions/median-of-two-sorted-arrays
        /*         4. Median of Two Sorted Arrays
        https://leetcode.com/problems/median-of-two-sorted-arrays/description
         */
        public class MedianOfTwoSortedArraysSol
        {
            //0. Naive - T:O(n+m) | s:O(1)
            public float MedianOfTwoSortedArraysNaive(int[] arrayOne, int[] arrayTwo)
            {
                int idxOne = 0, idxTwo = 0;
                int midIdx = (arrayOne.Length + arrayTwo.Length - 1) / 2;

                while (idxOne + idxTwo < midIdx)
                {
                    if (idxOne >= arrayOne.Length)
                        idxTwo++;
                    else if (idxTwo >= arrayTwo.Length)
                    {
                        idxOne++;
                    }
                    else if (arrayOne[idxOne] < arrayTwo[idxTwo])
                    {
                        idxOne++;
                    }
                    else idxTwo++;
                }
                int valueOne, valueTwo;
                if (arrayOne.Length + arrayTwo.Length % 2 == 0)
                { // case of even number size

                    bool areBothValuesArrayOne = idxTwo >= arrayTwo.Length || (idxOne + 1 < arrayOne.Length && arrayTwo[idxTwo] > arrayOne[idxOne + 1]);
                    bool areBothValuesArrayTwo = idxOne >= arrayOne.Length || (idxTwo + 1 < arrayTwo.Length && arrayOne[idxOne] > arrayTwo[idxTwo + 1]);

                    valueOne = areBothValuesArrayOne ? arrayOne[idxOne + 1] : arrayTwo[idxTwo];
                    valueTwo = areBothValuesArrayTwo ? arrayTwo[idxTwo + 1] : arrayOne[idxOne];

                    return (float)(valueOne + valueTwo) / 2;
                }
                valueOne = idxOne < arrayOne.Length ? arrayOne[idxOne] : Int32.MaxValue;
                valueTwo = idxTwo < arrayTwo.Length ? arrayTwo[idxTwo] : Int32.MaxValue;

                return Math.Min(valueOne, valueTwo);
            }

            int p1 = 0, p2 = 0;
            /*
            Approach 1: Merge Sort
Complexity Analysis
Let m be the size of array nums1 and n be the size of array nums2.
•	Time complexity: O(m+n)
o	We get the smallest element by comparing two values at p1 and p2, it takes O(1) to compare two elements and move the corresponding pointer to the right.
o	We need to traverse half of the arrays before reaching the median element(s).
o	To sum up, the time complexity is O(m+n).
•	Space complexity: O(1)
o	We only need to maintain two pointers p1 and p2.

            */
            public double UsingMergeSort(int[] nums1, int[] nums2)
            {
                int m = nums1.Length, n = nums2.Length;
                if ((m + n) % 2 == 0)
                {
                    for (int i = 0; i < (m + n) / 2 - 1; ++i)
                    {
                        int tmp = GetMin(nums1, nums2);
                    }

                    return (double)(GetMin(nums1, nums2) + GetMin(nums1, nums2)) / 2;
                }
                else
                {
                    for (int i = 0; i < (m + n) / 2; ++i)
                    {
                        int tmp = GetMin(nums1, nums2);
                    }

                    return GetMin(nums1, nums2);
                }
            }
            private int GetMin(int[] nums1, int[] nums2)
            {
                if (p1 < nums1.Length && p2 < nums2.Length)
                {
                    return nums1[p1] < nums2[p2] ? nums1[p1++] : nums2[p2++];
                }
                else if (p1 < nums1.Length)
                {
                    return nums1[p1++];
                }
                else if (p2 < nums2.Length)
                {
                    return nums2[p2++];
                }

                return -1;
            }
            /*
            Approach 2: Binary Search, Recursive
Complexity Analysis
Let m be the size of array nums1 and n be the size of array nums2.
•	Time complexity: O(log(m⋅n))
o	At each step, we cut one half off from either nums1 or nums2. If one of the arrays is emptied, we can directly get the target from the other array in a constant time. Therefore, the total time spent depends on when one of the arrays is cut into an empty array.
o	In the worst-case scenario, we may need to cut both arrays before finding the target element.
o	One of the two arrays is cut in half at each step, thus it takes logarithmic time to empty an array. The time to empty two arrays are independent of each other.
o	Therefore, the time complexity is O(logm+logn).
O(logm+logn)=O(log(m⋅n))
•	Space complexity: O(logm+logn)
o	Similar to the analysis on time complexity, the recursion steps depend on the number of iterations before we cut an array into an empty array. In the worst-case scenario, we need O(logm+logn) recursion steps.
o	However, during the recursive self-call, we only need to maintain 4 pointers: a_start, a_end, b_start and b_end. The last step of the function is to call itself, so if tail call optimization is implemented, the call stack always has O(1) records.
o	Please refer to Tail Call for more information on tail call optimization.

            */
            public double BinarySearchRec(int[] A, int[] B)
            {
                int na = A.Length, nb = B.Length;
                int n = na + nb;
                if ((na + nb) % 2 == 1)
                {
                    return Solve(A, B, n / 2, 0, na - 1, 0, nb - 1);
                }
                else
                {
                    return (double)(Solve(A, B, n / 2, 0, na - 1, 0, nb - 1) +
                                    Solve(A, B, n / 2 - 1, 0, na - 1, 0, nb - 1)) /
                           2;
                }
            }

            private int Solve(int[] A, int[] B, int k, int aStart, int aEnd, int bStart,
                             int bEnd)
            {
                // If the segment of on array is empty, it means we have passed all
                // its element, just return the corresponding element in the other
                // array.
                if (aEnd < aStart)
                {
                    return B[k - aStart];
                }

                if (bEnd < bStart)
                {
                    return A[k - bStart];
                }

                // Get the middle indexes and middle values of A and B.
                int aIndex = (aStart + aEnd) / 2, bIndex = (bStart + bEnd) / 2;
                int aValue = A[aIndex], bValue = B[bIndex];

                // If k is in the right half of A + B, remove the smaller left half.
                if (aIndex + bIndex < k)
                {
                    if (aValue > bValue)
                    {
                        return Solve(A, B, k, aStart, aEnd, bIndex + 1, bEnd);
                    }
                    else
                    {
                        return Solve(A, B, k, aIndex + 1, aEnd, bStart, bEnd);
                    }
                }
                // Otherwise, remove the larger right half.
                else
                {
                    if (aValue > bValue)
                    {
                        return Solve(A, B, k, aStart, aIndex - 1, bStart, bEnd);
                    }
                    else
                    {
                        return Solve(A, B, k, aStart, aEnd, bStart, bIndex - 1);
                    }
                }
            }
            /*
            Approach 3: A Better Binary Search
            Complexity Analysis
            Let m be the size of array nums1 and n be the size of array nums2.
            •	Time complexity: O(log(min(m,n)))
            o	We perform a binary search over the smaller array of size min(m,n).
            •	Space complexity: O(1)
            o	The algorithm only requires a constant amount of additional space to store and update a few parameters during the binary search.

            */
            public double FindMedianSortedArrays(int[] nums1, int[] nums2)
            {
                if (nums1.Length > nums2.Length)
                {
                    return FindMedianSortedArrays(nums2, nums1);
                }

                int m = nums1.Length, n = nums2.Length;
                int left = 0, right = m;

                while (left <= right)
                {
                    int partitionA = (left + right) / 2;
                    int partitionB = (m + n + 1) / 2 - partitionA;

                    int maxLeftA =
                        (partitionA == 0) ? int.MinValue : nums1[partitionA - 1];
                    int minRightA =
                        (partitionA == m) ? int.MaxValue : nums1[partitionA];
                    int maxLeftB =
                        (partitionB == 0) ? int.MinValue : nums2[partitionB - 1];
                    int minRightB =
                        (partitionB == n) ? int.MaxValue : nums2[partitionB];

                    if (maxLeftA <= minRightB && maxLeftB <= minRightA)
                    {
                        if ((m + n) % 2 == 0)
                        {
                            return (Math.Max(maxLeftA, maxLeftB) +
                                    Math.Min(minRightA, minRightB)) /
                                   2.0;
                        }
                        else
                        {
                            return Math.Max(maxLeftA, maxLeftB);
                        }
                    }
                    else if (maxLeftA > minRightB)
                    {
                        right = partitionA - 1;
                    }
                    else
                    {
                        left = partitionA + 1;
                    }
                }

                return 0.0;
            }
        }

        //https://www.algoexpert.io/questions/sort-k-sorted-array
        public static int[] SortedKSortedArray(int[] array, int k)
        {

            //1.T:O(n*k) | S:O(n) - using additional arry and loop to k+1 for each i

            //2.Optimal using MinHeap=> T:O(nlog(k)) | S:O(k)
            return SortedKSortedArrayUsingMinHeap(array, k);
        }

        private static int[] SortedKSortedArrayUsingMinHeap(int[] array, int k)
        {
            List<int> heapValues = new List<int>();
            for (int i = 0; i < Math.Min(k + 1, array.Length); i++)
                heapValues.Add(array[i]);

            Heap<int> minHeap = new Heap<int>(heapValues, Heap<int>.MIN_HEAP_FUNC); //O(n)
            int nextIndexToInsertElement = 0;
            for (int i = k + 1; i < array.Length; i++) //O(n)
            {
                int minVal = minHeap.Remove(); //O(log(k))

                if (array[nextIndexToInsertElement] != minVal) array[nextIndexToInsertElement] = minVal;

                nextIndexToInsertElement += 1;

                minHeap.Insert(array[i]);//O(log(k))


            }
            while (!minHeap.IsEmpty())
            {
                int minValue = minHeap.Remove();
                array[nextIndexToInsertElement] = minValue;
                nextIndexToInsertElement += 1;
            }

            return array;
        }

        //https://www.algoexpert.io/questions/move-element-to-end
        public static List<int> MoveElementToEnd(List<int> array, int toMove)
        {

            //1.Naive- Sort the array and do swaping until all toMove to end
            //T:O(nlogn) | S:O(1)

            //2.Optimal using Two pointer approach
            //T:O(n) | S:O(1)
            int leftIdx = 0, rightIdx = array.Count - 1;

            while (leftIdx < rightIdx)
            {

                while (leftIdx < rightIdx && array[rightIdx] == toMove)
                {
                    rightIdx--;
                }
                if (array[leftIdx] == toMove)
                {
                    int temp = array[rightIdx];
                    array[rightIdx] = array[leftIdx];
                    array[leftIdx] = temp;
                    leftIdx++;
                }

            }
            // Write your code here.
            return array;
        }
        //https://www.algoexpert.io/questions/largest-range
        public static int[] LargestRange(int[] array)
        {
            //1.Naive- Use Sort and check sequence to find out range
            //T:O(nlogn) | S:O(1)

            //2.Optimal- using extra space
            //T:O(n) | S:O(n)

            return LargestRangeOptimal(array);

        }

        private static int[] LargestRangeOptimal(int[] array)
        {
            int[] bestRange = new int[2];
            int longestLength = 0;
            HashSet<int> nums = new HashSet<int>();
            foreach (int num in array)
                nums.Add(num);

            foreach (int num in array)
            {
                if (!nums.Contains(num))
                    continue;

                nums.Remove(num);

                int curLen = 1;
                int left = num - 1;
                int right = num + 1;

                while (nums.Contains(left))
                {
                    nums.Remove(left);
                    curLen++;
                    left--;
                }
                while (nums.Contains(right))
                {
                    nums.Remove(right);
                    curLen++;
                    right++;
                }
                if (curLen > longestLength)
                {
                    longestLength = curLen;
                    bestRange = new int[] { left + 1, right - 1 };
                }

            }
            return bestRange;
        }
        //https://www.algoexpert.io/questions/array-of-products
        public static int[] ArrayOfProducts(int[] array)
        {

            //1. Naive 
            //T:O(n^2) | S:O(n)
            int[] products = ArrayOfProductsNaive(array);

            //2. Optimal with two arrays
            //T:O(n) | S:O(n)
            products = ArrayOfProductsOptimal1(array);

            //2. Optimal with single array
            //T:O(n) | S:O(n)
            products = ArrayOfProductsOptimal2(array);
            return products;

        }

        private static int[] ArrayOfProductsOptimal2(int[] array)
        {
            int[] products = new int[array.Length];

            int leftRunningProduct = 1;
            for (int i = 0; i < array.Length; i++)
            {
                products[i] = leftRunningProduct;
                leftRunningProduct *= array[i];
            }

            int rightRunningProduct = 1;
            for (int i = 0; i < array.Length; i++)
            {
                products[i] *= rightRunningProduct;
                rightRunningProduct *= array[i];
            }

            return products;
        }

        private static int[] ArrayOfProductsOptimal1(int[] array)
        {
            int[] products = new int[array.Length];
            int[] leftProducts = new int[array.Length];
            int[] rightProducts = new int[array.Length];

            int leftRunningProduct = 1;
            for (int i = 0; i < array.Length; i++)
            {
                leftProducts[i] = leftRunningProduct;
                leftRunningProduct *= array[i];
            }

            int rightRunningProduct = 1;
            for (int i = array.Length - 1; i >= 0; i--)
            {

                rightProducts[i] = rightRunningProduct;
                rightRunningProduct *= array[i];

            }
            for (int i = 0; i < array.Length; i++)
            {
                products[i] = leftProducts[i] * rightProducts[i];
            }
            return products;

        }

        private static int[] ArrayOfProductsNaive(int[] array)
        {
            int[] products = new int[array.Length];
            for (int i = 0; i < array.Length; i++)
            {
                int runningProduct = 1;
                for (int j = 0; j < array.Length; j++)
                {
                    if (i != j)
                    {
                        runningProduct *= array[j];
                    }
                }
                products[i] = runningProduct;
            }
            return products;
        }

        //https://www.algoexpert.io/questions/product-sum
        public static int ProductSum(List<object> array)
        {
            //T:O(n) | S:O(d) - n is total number of elements in array icluding sub array and d is greatest depth of special arrays in the array
            return ProductSumRec(array, 1);
        }
        private static int ProductSumRec(List<object> array, int multiplier)
        {
            int sum = 0;
            foreach (object el in array)
            {
                if (el is IList<object>)
                {
                    sum += ProductSumRec((List<object>)el, multiplier + 1);
                }
                else sum += (int)el;

            }
            return sum * multiplier;
        }




        //https://www.algoexpert.io/questions/three-number-sum
        public static List<int[]> ThreeNumberSum(int[] array, int targetSum)
        {
            Array.Sort(array);
            List<int[]> triplets = new List<int[]>();

            for (int i = 0; i < array.Length - 2; i++)
            {

                int leftPtr = i + 1, rightPtr = array.Length - 1;

                while (leftPtr < rightPtr)
                {

                    int currSum = array[i] + array[leftPtr] + array[rightPtr];

                    if (currSum == targetSum)
                    {
                        triplets.Add(new int[] { array[i], array[leftPtr], array[rightPtr] });
                        leftPtr++;
                        rightPtr--;
                    }
                    else if (currSum < targetSum)
                    {
                        leftPtr++;
                    }
                    else
                    {  // currSum > targetSum
                        rightPtr--;
                    }
                }
            }
            return triplets;
        }
        //https://www.algoexpert.io/questions/three-number-sort
        public static int[] ThreeNumberSort(int[] array, int[] order)
        {

        TODO:
            //1.Naive - modified bucket sort with multiple passed of array
            //T:O(n) | S:O(1)
            array = ThreeNumberSortNaive(array, order);

            //2.Optimal - two pointers with two passes of array
            //T:O(n) | S:O(1)
            array = ThreeNumberSortOptimal(array, order);

            //3.Optimal - three pointers with one pass of array
            //T:O(n) | S:O(1)
            array = ThreeNumberSortOptimal1(array, order);

            return array;

        }

        private static int[] ThreeNumberSortOptimal1(int[] array, int[] order)
        {
            int firstVal = order[0];
            int secondVal = order[1];

            int firstIdx = 0;
            int secondIdx = 0;
            int thridIdx = array.Length - 1;

            while (secondIdx <= thridIdx)
            {
                int value = array[secondIdx];
                if (value == firstVal)
                {
                    Swap(firstIdx, secondIdx, array);
                    firstIdx += 1;
                    secondIdx += 1;
                }
                else if (value == secondVal)
                {
                    secondIdx += 1;
                }
                else
                {
                    Swap(secondIdx, thridIdx, array);
                    thridIdx -= 1;
                }
            }
            return array;
        }

        private static int[] ThreeNumberSortOptimal(int[] array, int[] order)
        {
            int firstVal = order[0];
            int thridVal = order[2];

            int firstIdx = 0;
            for (int idx = 0; idx < array.Length; idx++)
            {
                if (array[idx] == firstVal)
                {
                    Swap(firstIdx, idx, array);
                    firstIdx += 1;
                }
            }
            int thridIdx = array.Length - 1;
            for (int idx = array.Length - 1; idx >= 0; idx--)
            {
                if (array[idx] == thridVal)
                {
                    Swap(thridIdx, idx, array);
                    thridIdx -= 1;
                }
            }
            return array;
        }

        private static void Swap(int i, int j, int[] array)
        {
            int temp = array[i];
            array[j] = array[i];
            array[i] = temp;
        }

        private static int[] ThreeNumberSortNaive(int[] array, int[] order)
        {
            int[] valueCounts = new int[] { 0, 0, 0 };

            foreach (var element in array)
            {
                int orderIdx = GetIndex(order, element);
                valueCounts[orderIdx] += 1;
            }
            for (int i = 0; i < 3; i++)
            {
                int value = order[i];
                int count = valueCounts[i];

                int numElementsBefore = GetSum(valueCounts, i);
                for (int n = 0; n < count; n++)
                {
                    int curIndx = numElementsBefore + n;
                    array[curIndx] = value;
                }
            }
            return array;
        }

        private static int GetSum(int[] valueCounts, int end)
        {
            int sum = 0;
            for (int i = 0; i < end; i++) sum += valueCounts[i];
            return sum;
        }

        private static int GetIndex(int[] order, int element)
        {
            for (int i = 0; i < order.Length; i++)
            {
                if (order[i] == element)
                    return i;
            }
            return -1;
        }
        //https://www.algoexpert.io/questions/numbers-in-pi
        public static int NumbersInPI(string pi, string[] numbers)
        {

            //1. T:O(n^3 + m) | S:O(n + m) - where n is the number of digits in Pi and m is the number of favorite numbers
            int result = NumbersInPI1(pi, numbers);

            //2. reverseal- T:O(n^3 + m) | S:O(n + m) - where n is the number of digits in Pi and m is the number of favorite numbers
            result = NumbersInPI2(pi, numbers);
            return result;

        }

        private static int NumbersInPI2(string pi, string[] numbers)
        {
            HashSet<string> numbersTable = new HashSet<string>();
            foreach (string number in numbers)
            {
                numbersTable.Add(number);
            }

            Dictionary<int, int> cache = new Dictionary<int, int>();
            for (int i = pi.Length - 1; i >= 0; i--)
            {
                GetMinSpaces(pi, numbersTable, cache, i);
            }

            return cache[0] == Int32.MaxValue ? -1 : cache[0];
        }

        private static int NumbersInPI1(string pi, string[] numbers)
        {
            HashSet<string> numbersTable = new HashSet<string>();
            foreach (string number in numbers)
            {
                numbersTable.Add(number);
            }

            Dictionary<int, int> cache = new Dictionary<int, int>();
            int minSpaces = GetMinSpaces(pi, numbersTable, cache, 0);
            return minSpaces == Int32.MaxValue ? -1 : minSpaces;
        }

        private static int GetMinSpaces(string pi, HashSet<string> numbersTable, Dictionary<int, int> cache, int idx)
        {
            if (idx == pi.Length) return -1;
            if (cache.ContainsKey(idx)) return cache[idx];
            int minSpaces = Int32.MaxValue;
            for (int i = idx; i < pi.Length; i++)
            {

                string prefix = pi.Substring(idx, i + 1 - idx); //O(n) as strings are immutable in C#
                if (numbersTable.Contains(prefix))
                {
                    int minSpacesInSuffix = GetMinSpaces(pi, numbersTable, cache, i + 1);
                    //Handle INT overflow
                    if (minSpacesInSuffix == Int32.MaxValue)
                    {
                        minSpaces = Math.Min(minSpaces, minSpacesInSuffix);
                    }
                    else
                    {
                        minSpaces = Math.Min(minSpaces, minSpacesInSuffix + 1);
                    }
                }
            }
            cache[idx] = minSpaces;
            return cache[idx];
        }
        //https://www.algoexpert.io/questions/smallest-difference
        public static int[] SmallestDiff(int[] arrayOne, int[] arrayTwo)
        {

            //1. Naive - Pair of loops
            //T:O(n^2) : S(1)

            //2. Sorting and leveraging sort properties
            //T:O(nlongn)+O(mlogm) | O(1)
            Array.Sort(arrayOne);
            Array.Sort(arrayTwo);

            int idxOne = 0, idxTwo = 0;
            int globalSmallestDiff = Int32.MaxValue;
            int runningDiff = Int32.MaxValue;
            int[] smallestDiffPair = new int[2];

            while (idxOne < arrayOne.Length && idxTwo < arrayTwo.Length)
            {
                int firstNum = arrayOne[idxOne];
                int secondNum = arrayTwo[idxTwo];

                if (firstNum < secondNum)
                {
                    runningDiff = secondNum - firstNum;
                    idxOne++;
                }
                else if (secondNum < firstNum)
                {
                    runningDiff = firstNum - secondNum;
                    idxTwo++;
                }
                else
                {
                    return new int[] { firstNum, secondNum };
                }
                if (runningDiff < globalSmallestDiff)
                {
                    globalSmallestDiff = runningDiff;
                    smallestDiffPair = new int[] { firstNum, secondNum };
                }

            }
            return smallestDiffPair;
        }
        public static int[] SmallestDiff2(int[] arrayOne, int[] arrayTwo)
        {
            Array.Sort(arrayOne);
            Array.Sort(arrayTwo);

            long[] smallestDiffPair = new long[] { Int32.MinValue, Int32.MaxValue };
            int idxOne = 0, idxTwo = 0;

            while (idxOne < arrayOne.Length && idxTwo < arrayTwo.Length)
            {

                int num1 = arrayOne[idxOne];
                int num2 = arrayTwo[idxTwo];

                if (Math.Abs(num1 - num2) < Math.Abs(smallestDiffPair[0] - smallestDiffPair[1]))
                {
                    smallestDiffPair = new long[] { num1, num2 };
                }

                if (num1 < num2)
                {
                    idxOne++;
                }
                else if (num1 > num2)
                {
                    idxTwo++;
                }
                else
                {
                    return new int[] { (int)smallestDiffPair[0], (int)smallestDiffPair[1] };
                }
            }
            return new int[] { (int)smallestDiffPair[0], (int)smallestDiffPair[1] };
        }

        //https://www.algoexpert.io/questions/monotonic-array
        public static bool IsMonotonic(int[] array)
        {
            //T:O(n)|S:O(1)
            var isNonDecreasing = true;
            var isNonIncreasing = true;

            for (int i = 1; i < array.Length; i++)
            {

                if (array[i] < array[i - 1]) isNonDecreasing = false;
                if (array[i] > array[i - 1]) isNonIncreasing = false;
            }

            return isNonDecreasing || isNonIncreasing;

        }
        //https://www.algoexpert.io/questions/first-duplicate-value
        public static int FirstDuplicateValue(int[] array)
        {

            //1.Naive - pair of loops
            //T:O(n^2) | O(1)
            int result = FirstDuplicateValueNaive(array);

            //2.Optimal - using Extra Space
            //T:O(n) | O(n)
            result = FirstDuplicateValueOptimal1(array);

            //3.Optimal - using problem data i.e array contains numbers between 1 to n, inclusive.
            //T:O(n) | O(n)
            result = FirstDuplicateValueOptimal2(array);

            return result;
        }

        private static int FirstDuplicateValueOptimal2(int[] array)
        {
            for (int i = 0; i < array.Length; i++)
            {

                int absVal = Math.Abs(array[i]);
                if (array[absVal - 1] < 0) return absVal;

                array[absVal - 1] *= -1;
            }
            return -1;
        }

        private static int FirstDuplicateValueOptimal1(int[] array)
        {
            HashSet<int> set = new HashSet<int>();
            foreach (var val in array)
            {
                if (set.Contains(val))
                    return val;
                set.Add(val);
            }
            return -1;
        }

        private static int FirstDuplicateValueNaive(int[] array)
        {
            int minIndex = array.Length;
            for (int i = 0; i < array.Length; i++)
            {

                int val = array[i];
                for (int j = i + 1; j < array.Length; j++)
                {

                    if (array[j] == val)
                    {
                        if (minIndex < j)
                        {
                            minIndex = j;
                        }
                    }
                }
            }
            return minIndex == array.Length ? -1 : array[minIndex];

        }
        //https://www.algoexpert.io/questions/majority-element
        public static int MajorityElement(int[] array)
        {

            //1.Naive - Pair of loops
            //T:O(n^2) | S:O(n)

            //2.Using sorting if allowed
            //T:O(nlogn) | S:O(1)

            //3.Using Dictionary
            //T:O(n) | S:O(n)

            //4.Optimal - Using logic/trick of assuming first element is a majority one.
            //T:O(n) | S:O(1)
            int majorityElement = MajorityElementOptimal1(array);

            //5.Optimal - using bit operations
            //T:O(n) | S:O(1)
            majorityElement = MajorityElementOptimal2(array);

            return majorityElement;

        }

        private static int MajorityElementOptimal2(int[] array)
        {
            int answer = 0;
            for (int curBit = 0; curBit < 32; curBit++)
            {
                int curBitVal = 1 << curBit;
                int onesCount = 0;

                foreach (var num in array)
                {
                    if ((num & curBitVal) != 0)
                        onesCount++;

                }
                if (onesCount > array.Length / 2)
                    answer += curBitVal;
            }
            return answer;
        }

        private static int MajorityElementOptimal1(int[] array)
        {
            int answer = array[0], count = 0;

            foreach (var val in array)
            {
                if (count == 0)
                    answer = val;

                if (val == answer) count++;
                else count--;

            }
            return answer;
        }

        /* 229. Majority Element II
        https://leetcode.com/problems/majority-element-ii/description/
         */
        class MjorityElementIISol
        {
            /*
            Approach 1: Boyer-Moore Voting Algorithm
                        Complexity Analysis
            •	Time complexity : O(N) where N is the size of nums. We first go through nums looking for first and second potential candidates. We then count the number of occurrences for these two potential candidates in nums. Therefore, our runtime is O(N)+O(N)=O(2N)≈O(N).
            •	Space complexity : O(1) since we only have four variables for holding two potential candidates and two counters. Even the returning array is at most 2 elements.
             */
            public List<int> UsingBoyerMooreVotingAlgo(int[] nums)
            {

                // 1st pass
                int count1 = 0;
                int count2 = 0;

                int? candidate1 = null;
                int? candidate2 = null;

                foreach (int n1 in nums)
                {
                    if (candidate1 != null && candidate1 == n1)
                    {
                        count1++;
                    }
                    else if (candidate2 != null && candidate2 == n1)
                    {
                        count2++;
                    }
                    else if (count1 == 0)
                    {
                        candidate1 = n1;
                        count1++;
                    }
                    else if (count2 == 0)
                    {
                        candidate2 = n1;
                        count2++;
                    }
                    else
                    {
                        count1--;
                        count2--;
                    }
                }

                // 2nd pass
                List<int> result = new();

                count1 = 0;
                count2 = 0;

                foreach (int n1 in nums)
                {
                    if (candidate1 != null && n1 == candidate1) count1++;
                    if (candidate2 != null && n1 == candidate2) count2++;
                }

                int n = nums.Length;
                if (count1 > n / 3) result.Add((int)candidate1);
                if (count2 > n / 3) result.Add((int)candidate2);

                return result;
            }
        }

        //https://www.algoexpert.io/questions/quickselect
        //kth smalles/ laragest element
        public static int FindKthSmallest(int[] array, int k)
        {

            //1.Naive..using pair of loops
            //T:O(n^2) | S:O(1)

            //2.Sorting 
            //T:O(nlogn) | S:O(1)

            //3.Minheap 
            //T:O(n) | S:O(n)

            //2.QuickSelect - Apt one
            //T:O(n) | S:O(1) - Best || T:O(n) | S:O(1) - Avg || T:O(n^2) | S:O(1) - Worst
            //Iterative approach is apt for QuickSelect than recursive  as it loop thru only one part of the arary and not two parts alike QuickSort
            //Also it comes with constant space
            int position = k - 1;
            return FindKthSmallestUsingQuickSelect(array, 0, array.Length - 1, position);

        }

        private static int FindKthSmallestUsingQuickSelect(int[] array, int startIdx, int endIdx, int position)
        {
            while (true)
            {

                if (startIdx > endIdx) throw new Exception("Your algo should never arrive here");

                int pivotIdx = startIdx;
                int leftIdx = startIdx + 1;
                int rightIdx = endIdx;

                while (leftIdx <= rightIdx)
                {

                    if (array[leftIdx] > array[pivotIdx] && array[rightIdx] < array[pivotIdx])
                    {
                        Swap(array, leftIdx, rightIdx);
                    }
                    if (array[leftIdx] <= array[pivotIdx]) leftIdx++;
                    if (array[rightIdx] >= array[pivotIdx]) rightIdx--;
                }
                Swap(array, rightIdx, pivotIdx);
                if (rightIdx == position)
                    return array[rightIdx];
                else if (rightIdx < position) startIdx = rightIdx + 1;
                else endIdx = rightIdx - 1;
            }

        }
        //https://www.algoexpert.io/questions/common-characters
        public static string[] CommonCharacters(string[] strings)
        {

            //1.More memory
            //T:O(n*m) | O(c) - n is number of strings and m is length of largest string and c is the number unique chars acorss all strings
            string[] commonChars = CommonCharacters1(strings);

            //2.Less memory
            //T:O(n*m) | O(m) - n is number of strings and m is length of largest string
            commonChars = CommonCharacters2(strings);

            return commonChars;

        }

        private static string[] CommonCharacters2(string[] strings)
        {
            string smallestString = GetSmalletString(strings);
            HashSet<char> potentialCommonChars = new HashSet<char>();
            foreach (var ch in smallestString)
            {
                potentialCommonChars.Add(ch);
            }
            foreach (string str in strings)
            {
                RemoveNoneexistentChars(str, potentialCommonChars);
            }
            string[] commonChars = new string[potentialCommonChars.Count];
            int j = 0;
            foreach (var ch in potentialCommonChars)
            {
                commonChars[j] = ch.ToString();
                j++;
            }
            return commonChars;
        }

        private static void RemoveNoneexistentChars(string str, HashSet<char> potentialCommonChars)
        {
            HashSet<char> uniqueChars = new HashSet<char>();
            foreach (var ch in str)
            {
                uniqueChars.Add(ch);
            }
            foreach (var ch in potentialCommonChars)
            {
                if (!uniqueChars.Contains(ch)) potentialCommonChars.Remove(ch);
            }
        }

        private static string GetSmalletString(string[] strings)
        {
            string smallestString = strings[0];
            foreach (var str in strings)
            {
                if (smallestString.Length > str.Length)
                    smallestString = str;
            }
            return smallestString;

        }

        private static string[] CommonCharacters1(string[] strings)
        {
            Dictionary<char, int> charCounts = new Dictionary<char, int>();
            foreach (string str in strings)
            {
                HashSet<char> uniqueChars = new HashSet<char>();
                foreach (char ch in str)
                {
                    uniqueChars.Add(ch);
                }
                foreach (char ch in uniqueChars)
                {
                    if (!charCounts.ContainsKey(ch))
                        charCounts[ch] = 0;
                    charCounts[ch] += 1;
                }

            }
            List<char> finalChars = new List<char>();
            foreach (var Item in charCounts)
            {
                char ch = Item.Key;
                int count = Item.Value;
                if (count == strings.Length) finalChars.Add(ch);

            }
            string[] finalCharArr = new string[finalChars.Count];
            for (int i = 0; i < finalChars.Count; i++)
            {
                finalCharArr[i] = finalChars[i].ToString();
            }

            return finalCharArr;
        }
        //https://www.algoexpert.io/questions/missingNumbers
        public static int[] MissingNumbers(int[] nums)
        {
            //1.Naive - Sorting
            //T:O(nlogn) | S:O(1)

            //2.Optimal - Using HashSet
            //T:O(n) | S:O(n)
            int[] missingNums = MissingNumbersOptimal1(nums);

            //3.Optimal - summing up all to n and in nums to deal with average
            //T:O(n) | S:O(1)
            missingNums = MissingNumbersOptimal2(nums);

        //4.Optimal - Bit manipulation - XOR and &
        //T:O(n) | S:O(1) 
        TODO:
            missingNums = MissingNumbersOptimal3(nums);

            return missingNums;
        }

        private static int[] MissingNumbersOptimal3(int[] nums)
        {
            int solutionXOR = 0;
            for (int i = 0; i < nums.Length + 3; i++)
            {
                solutionXOR ^= i;
                if (i < nums.Length)
                {
                    solutionXOR ^= nums[i];
                }
            }
            int[] solution = new int[2];
            int setBit = solutionXOR & -solutionXOR;
            for (int i = 0; i < nums.Length + 3; i++)
            {
                if ((i & setBit) == 0)
                {
                    solution[0] ^= i;
                }
                else solution[1] ^= i;

                if (i < nums.Length)
                {
                    if ((nums[i] & setBit) == 0)
                    {
                        solution[0] ^= nums[i];

                    }
                    else solution[1] ^= nums[i];

                }
            }
            Array.Sort(solution);
            return solution;

        }

        private static int[] MissingNumbersOptimal2(int[] nums)
        {
            int total = 0;
            for (int num = 1; num < nums.Length + 3; num++)
            {
                total += num;
            }
            foreach (int num in nums)
            {
                total -= num;
            }

            int avgMissingVal = total / 2;

            int foundFirstHalf = 0;
            int foundSecondHalf = 0;

            foreach (var num in nums)
            {
                if (num <= avgMissingVal)
                {
                    foundFirstHalf += num;
                }
                else
                {
                    foundSecondHalf = +num;
                }
            }
            int expectedFirstHalf = 0;
            for (int i = 1; i <= avgMissingVal; i++)
            {
                expectedFirstHalf += i;
            }
            int expectedSecondHalf = 0;
            for (int i = avgMissingVal + 1; i < nums.Length + 3; i++)
            {
                expectedSecondHalf += i;
            }

            return new int[] { expectedFirstHalf - foundFirstHalf, expectedSecondHalf - foundSecondHalf };
        }

        private static int[] MissingNumbersOptimal1(int[] nums)
        {
            HashSet<int> includedNums = new HashSet<int>();
            foreach (int num in nums)
            {
                includedNums.Add(num);
            }
            int[] solution = new int[] { -1, -1 };
            for (int num = 1; num < nums.Length + 3; num++)
            {

                if (!includedNums.Contains(num))
                {
                    if (solution[0] == -1)
                    {
                        solution[0] = num;
                    }
                    else solution[1] = num;
                }
            }
            return solution;
        }
        //https://www.algoexpert.io/questions/subarray-sort
        // O(n) time | O(1) space
        public static int[] SubarraySort(int[] array)
        {
            int minOutOfOrder = Int32.MaxValue;
            int maxOutOfOrder = Int32.MinValue;
            for (int i = 0; i < array.Length; i++)
            {
                int num = array[i];
                if (IsOutOfOrder(i, num, array))
                {
                    minOutOfOrder = Math.Min(minOutOfOrder, num);
                    maxOutOfOrder = Math.Max(maxOutOfOrder, num);
                }
            }
            if (minOutOfOrder == Int32.MaxValue)
            {
                return new int[] { -1, -1 };
            }
            int subarrayLeftIdx = 0;
            while (minOutOfOrder >= array[subarrayLeftIdx])
            {
                subarrayLeftIdx++;
            }
            int subarrayRightIdx = array.Length - 1;
            while (maxOutOfOrder <= array[subarrayRightIdx])
            {
                subarrayRightIdx--;
            }
            return new int[] { subarrayLeftIdx, subarrayRightIdx };
        }

        public static bool IsOutOfOrder(int i, int num, int[] array)
        {
            if (i == 0)
            {
                return num > array[i + 1];
            }
            if (i == array.Length - 1)
            {
                return num < array[i - 1];
            }
            return num > array[i + 1] || num < array[i - 1];
        }

        //https://www.algoexpert.io/questions/next-greater-element
        // O(n) time | O(n) space - where n is the length of the array
        public int[] NextGreaterElement(int[] array)
        {
            int[] result = new int[array.Length];
            Array.Fill(result, -1);

            Stack<int> stack = new Stack<int>();

            for (int idx = 0; idx < 2 * array.Length; idx++)
            {
                int circularIdx = idx % array.Length;

                while (stack.Count > 0 && array[stack.Peek()] < array[circularIdx])
                {
                    int top = stack.Pop();
                    result[top] = array[circularIdx];
                }

                stack.Push(circularIdx);
            }

            return result;
        }
        // O(n) time | O(n) space - where n is the length of the array
        public int[] NextGreaterElement2(int[] array)
        {
            int[] result = new int[array.Length];
            Array.Fill(result, -1);

            Stack<int> stack = new Stack<int>();

            for (int idx = 2 * array.Length - 1; idx >= 0; idx--)
            {
                int circularIdx = idx % array.Length;

                while (stack.Count > 0)
                {
                    if (stack.Peek() <= array[circularIdx])
                    {
                        stack.Pop();
                    }
                    else
                    {
                        result[circularIdx] = stack.Peek();
                        break;
                    }
                }

                stack.Push(array[circularIdx]);
            }

            return result;
        }

        /*         503. Next Greater Element II
        https://leetcode.com/problems/next-greater-element-ii/description/ 
         */
        public class NextGreaterElementIISol
        {
            /* 
            Approach 1: Brute Force (using Double Length Array) [Time Limit Exceeded
            Complexity Analysis
            •	Time complexity : O(n^2). The complete doublenums array(of size 2n) is scanned for all the elements of nums in the worst case.
            •	Space complexity : O(n). doublenums array of size 2n is used. res array of size n is used.
             */
            public int[] Naive(int[] nums)
            {
                int[] res = new int[nums.Length];
                int[] doublenums = new int[nums.Length * 2];
                Array.Copy(nums, 0, doublenums, 0, nums.Length);
                Array.Copy(nums, 0, doublenums, nums.Length, nums.Length);
                for (int i = 0; i < nums.Length; i++)
                {
                    res[i] = -1;
                    for (int j = i + 1; j < doublenums.Length; j++)
                    {
                        if (doublenums[j] > doublenums[i])
                        {
                            res[i] = doublenums[j];
                            break;
                        }
                    }
                }
                return res;
            }
            /* Approach 2: Better Brute Force [Accepted]	
Complexity Analysis
•	Time complexity : O(n^2). The complete nums array of size n is scanned for all the elements of nums in the worst case.
•	Space complexity : O(n). res array of size n is used.
             */
            public int[] NaiveOptimal(int[] nums)
            {
                int[] res = new int[nums.Length];
                for (int i = 0; i < nums.Length; i++)
                {
                    res[i] = -1;
                    for (int j = 1; j < nums.Length; j++)
                    {
                        if (nums[(i + j) % nums.Length] > nums[i])
                        {
                            res[i] = nums[(i + j) % nums.Length];
                            break;
                        }
                    }
                }
                return res;
            }
            /* Approach 3: Using Stack [Accepted] 
        Complexity Analysis
•	Time complexity : O(n). Only two traversals of the nums array are done. Further, at most 2n elements are pushed and popped from the stack.
•	Space complexity : O(n). A stack of size n is used. res array of size n is used.
            */
            public int[] UsingStack(int[] nums)
            {
                int[] res = new int[nums.Length];
                Stack<int> stack = new Stack<int>();
                for (int i = 2 * nums.Length - 1; i >= 0; --i)
                {
                    while (stack.Count > 0 && nums[stack.Peek()] <= nums[i % nums.Length])
                    {
                        stack.Pop();
                    }
                    res[i % nums.Length] = stack.Count == 0 ? -1 : nums[stack.Peek()];
                    stack.Push(i % nums.Length);
                }
                return res;
            }

        }

        /*    556. Next Greater Element III 
        https://leetcode.com/problems/next-greater-element-iii/
         */
        public class NextGreaterElementIIISol
        {

            /* Approach #1 Brute Force
            Complexity Analysis
•	Time complexity : O(n!). A total of n! permutations are possible for a number consisting of n digits.
•	Space complexity : O(n!). A total of n! permutations are possible for a number consisting of n digits, with each permutation consisting of n digits.

             */
            public int Naive(int n)
            {
                String s = "" + n;
                Permute(s, 0, s.Length - 1);
                list.Sort();
                int i;
                for (i = list.Count - 1; i >= 0; i--)
                {
                    if (list[i].Equals("" + n))
                        break;
                }
                return i == list.Count - 1 ? -1 : int.Parse(list[i + 1]);
            }
            private String Swap(String s, int i0, int i1)
            {
                if (i0 == i1)
                    return s;
                String s1 = s.Substring(0, i0);
                String s2 = s.Substring(i0 + 1, i1);
                String s3 = s.Substring(i1 + 1);
                return s1 + s[i1] + s2 + s[i0] + s3;
            }
            List<String> list = new();
            private void Permute(String a, int l, int r)
            {
                int i;
                if (l == r)
                    list.Add(a);
                else
                {
                    for (i = l; i <= r; i++)
                    {
                        a = Swap(a, l, i);
                        Permute(a, l + 1, r);
                        a = Swap(a, l, i);
                    }
                }
            }
            /* Approach #2 Linear Solution
            Complexity Analysis
    •	Time complexity : O(n). In worst case, only two scans of the whole array are needed. Here, n refers to the number of digits in the given number.
    •	Space complexity : O(n). An array a of size n is used, where n is the number of digits in the given number.

             */
            public int UsingLinearSolution(int n)
            {
                char[] a = ("" + n).ToCharArray();
                int i = a.Length - 2;
                while (i >= 0 && a[i + 1] <= a[i])
                {
                    i--;
                }
                if (i < 0)
                    return -1;
                int j = a.Length - 1;
                while (j >= 0 && a[j] <= a[i])
                {
                    j--;
                }
                Swap(a, i, j);
                Reverse(a, i + 1);
                try
                {
                    return int.Parse(new String(a));
                }
                catch (Exception e)
                {
                    return -1;
                }
            }
            private void Reverse(char[] a, int start)
            {
                int i = start, j = a.Length - 1;
                while (i < j)
                {
                    Swap(a, i, j);
                    i++;
                    j--;
                }
            }
            private void Swap(char[] a, int i, int j)
            {
                char temp = a[i];
                a[i] = a[j];
                a[j] = temp;
            }

        }


        //https://www.algoexpert.io/questions/sorted-squared-
        // O(nlogn) time | O(n) space - where n is the length of the input array
        public int[] SortedSquaredArrayNaive(int[] array)
        {
            int[] sortedSquares = new int[array.Length];
            for (int idx = 0; idx < array.Length; idx++)
            {
                int value = array[idx];
                sortedSquares[idx] = value * value;
            }
            Array.Sort(sortedSquares);
            return sortedSquares;
        }
        // O(n) time | O(n) space - where n is the length of the input array
        public int[] SortedSquaredArrayOptimal(int[] array)
        {
            int[] sortedSquares = new int[array.Length];
            int smallerValueIdx = 0;
            int largerValueIdx = array.Length - 1;
            for (int idx = array.Length - 1; idx >= 0; idx--)
            {
                int smallerValue = array[smallerValueIdx];
                int largerValue = array[largerValueIdx];
                if (Math.Abs(smallerValue) > Math.Abs(largerValue))
                {
                    sortedSquares[idx] = smallerValue * smallerValue;
                    smallerValueIdx++;
                }
                else
                {
                    sortedSquares[idx] = largerValue * largerValue;
                    largerValueIdx--;
                }
            }
            return sortedSquares;
        }
        //https://www.algoexpert.io/questions/non-constructible-change
        // O(nlogn) time | O(1) space - where n is the number of coins
        public int NonConstructibleChange(int[] coins)
        {
            Array.Sort(coins);

            int currentChangeCreated = 0;
            foreach (var coin in coins)
            {
                if (coin > currentChangeCreated + 1)
                {
                    return currentChangeCreated + 1;
                }

                currentChangeCreated += coin;
            }

            return currentChangeCreated + 1;
        }
        //https://www.algoexpert.io/questions/find-three-largest-numbers
        // O(n) time | O(1) space
        public static int[] FindThreeLargestNumbers(int[] array)
        {
            /*
             Can we just sort the input array and return the last three elements in the sorted array?
            The best sorting algorithms run, on average, in O(nlog(n)) time; 
            we can implement a more optimal, linear-time algorithm (an O(n)-time algorithm) by not sorting the input array.

            */
            int[] threeLargest = { Int32.MinValue, Int32.MinValue, Int32.MinValue };
            foreach (int num in array)
            {
                updateLargest(threeLargest, num);
            }
            return threeLargest;
        }

        public static void updateLargest(int[] threeLargest, int num)
        {
            if (num > threeLargest[2])
            {
                shiftAndUpdate(threeLargest, num, 2);
            }
            else if (num > threeLargest[1])
            {
                shiftAndUpdate(threeLargest, num, 1);
            }
            else if (num > threeLargest[0])
            {
                shiftAndUpdate(threeLargest, num, 0);
            }
        }

        public static void shiftAndUpdate(int[] array, int num, int idx)
        {
            for (int i = 0; i <= idx; i++)
            {
                if (i == idx)
                {
                    array[i] = num;
                }
                else
                {
                    array[i] = array[i + 1];
                }
            }
        }
        //https://www.algoexpert.io/questions/minimum-characters-for-words
        // O(n * l) time | O(c) space - where n is the number of words,
        // l is the length of the longest word, and c is the number of
        // unique characters across all words
        // See notes under video explanation for details about the space complexity.
        public char[] MinimumCharactersForWords(string[] words)
        {
            Dictionary<char, int> maximumCharFrequencies = new Dictionary<char, int>();

            foreach (var word in words)
            {
                Dictionary<char, int> characterFrequencies = countCharFrequencies(word);
                updateMaximumFrequencies(characterFrequencies, maximumCharFrequencies);
            }

            return makeArrayFromCharFrequencies(maximumCharFrequencies);
        }

        public Dictionary<char, int> countCharFrequencies(string str)
        {
            Dictionary<char, int> characterFrequencies = new Dictionary<char, int>();

            foreach (var character in str.ToCharArray())
            {
                characterFrequencies[character] =
                  characterFrequencies.GetValueOrDefault(character, 0) + 1;
            }

            return characterFrequencies;
        }
        public void updateMaximumFrequencies(
          Dictionary<char, int> frequencies, Dictionary<char, int> maximumFrequencies
        )
        {
            foreach (var frequency in frequencies)
            {
                char character = frequency.Key;
                int characterFrequency = frequency.Value;

                if (maximumFrequencies.ContainsKey(character))
                {
                    maximumFrequencies[character] =
                      Math.Max(characterFrequency, maximumFrequencies[character]);
                }
                else
                {
                    maximumFrequencies[character] = characterFrequency;
                }
            }
        }
        public char[] makeArrayFromCharFrequencies(
          Dictionary<char, int> characterFrequencies
        )
        {
            List<char> characters = new List<char>();

            foreach (var frequency in characterFrequencies)
            {
                char character = frequency.Key;
                int characterFrequency = frequency.Value;

                for (int idx = 0; idx < characterFrequency; idx++)
                {
                    characters.Add(character);
                }
            }

            char[] charactersArray = new char[characters.Count];
            for (int idx = 0; idx < characters.Count; idx++)
            {
                charactersArray[idx] = characters[idx];
            }

            return charactersArray;
        }

        //https://www.algoexpert.io/questions/merge-sorted-arrays
        //1.
        // O(nk) time | O(n + k) space - where n is the total
        // number of array elements and k is the number of arrays
        public static List<int> MergeSortedArraysNaive(List<List<int>> arrays)
        {
            List<int> sortedList = new List<int>();
            List<int> elementIdxs = Enumerable.Repeat(0, arrays.Count).ToList();
            while (true)
            {
                List<Item> smallestItems = new List<Item>();
                for (int arrayIdx = 0; arrayIdx < arrays.Count; arrayIdx++)
                {
                    List<int> relevantArray = arrays[arrayIdx];
                    int elementIdx = elementIdxs[arrayIdx];
                    if (elementIdx == relevantArray.Count) continue;
                    smallestItems.Add(new Item(arrayIdx, relevantArray[elementIdx]));
                }
                if (smallestItems.Count == 0) break;
                Item nextItem = getMinValue(smallestItems);
                sortedList.Add(nextItem.num);
                elementIdxs[nextItem.arrayIdx] = elementIdxs[nextItem.arrayIdx] + 1;
            }

            return sortedList;
        }

        public static Item getMinValue(List<Item> items)
        {
            int minValueIdx = 0;
            for (int i = 1; i < items.Count; i++)
            {
                if (items[i].num < items[minValueIdx].num) minValueIdx = i;
            }
            return items[minValueIdx];
        }

        public class Item
        {
            public int arrayIdx;
            public int elementIdx;
            public int num;

            public Item(int arrayIdx, int num)
            {
                this.arrayIdx = arrayIdx;
                this.num = num;

            }
            public Item(int arrayIdx, int elementIdx, int num)
            {
                this.arrayIdx = arrayIdx;
                this.elementIdx = elementIdx;
                this.num = num;
            }
        }

        // O(nlog(k) + k) time | O(n + k) space - where n is the total
        // number of array elements and k is the number of arrays
        public static List<int> MergeSortedArraysOptimal(List<List<int>> arrays)
        {
            List<int> sortedList = new List<int>();
            List<Item> smallestItems = new List<Item>();

            for (int arrayIdx = 0; arrayIdx < arrays.Count; arrayIdx++)
            {
                smallestItems.Add(new Item(arrayIdx, 0, arrays[arrayIdx][0]));
            }

            MinHeap minHeap = new MinHeap(smallestItems);
            while (!minHeap.isEmpty())
            {
                Item smallestItem = minHeap.Remove();
                sortedList.Add(smallestItem.num);
                if (smallestItem.elementIdx == arrays[smallestItem.arrayIdx].Count - 1)
                    continue;
                minHeap.Insert(new Item(
                  smallestItem.arrayIdx,
                  smallestItem.elementIdx + 1,
                  arrays[smallestItem.arrayIdx][smallestItem.elementIdx + 1]
                ));
            }

            return sortedList;
        }

        public class MinHeap
        {
            List<Item> heap = new List<Item>();

            public MinHeap(List<Item> array)
            {
                heap = buildHeap(array);
            }

            public bool isEmpty()
            {
                return heap.Count == 0;
            }

            public List<Item> buildHeap(List<Item> array)
            {
                int firstParentIdx = (array.Count - 2) / 2;
                for (int currentIdx = firstParentIdx; currentIdx >= 0; currentIdx--)
                {
                    siftDown(currentIdx, array.Count - 1, array);
                }
                return array;
            }

            public void siftDown(int currentIdx, int endIdx, List<Item> heap)
            {
                int childOneIdx = currentIdx * 2 + 1;
                while (childOneIdx <= endIdx)
                {
                    int childTwoIdx =
                      currentIdx * 2 + 2 <= endIdx ? currentIdx * 2 + 2 : -1;
                    int idxToSwap;
                    if (childTwoIdx != -1 && heap[childTwoIdx].num < heap[childOneIdx].num)
                    {
                        idxToSwap = childTwoIdx;
                    }
                    else
                    {
                        idxToSwap = childOneIdx;
                    }
                    if (heap[idxToSwap].num < heap[currentIdx].num)
                    {
                        swap(currentIdx, idxToSwap, heap);
                        currentIdx = idxToSwap;
                        childOneIdx = currentIdx * 2 + 1;
                    }
                    else
                    {
                        return;
                    }
                }
            }

            public void siftUp(int currentIdx, List<Item> heap)
            {
                int parentIdx = (currentIdx - 1) / 2;
                while (currentIdx > 0 && heap[currentIdx].num < heap[parentIdx].num)
                {
                    swap(currentIdx, parentIdx, heap);
                    currentIdx = parentIdx;
                    parentIdx = (currentIdx - 1) / 2;
                }
            }

            public Item Remove()
            {
                swap(0, heap.Count - 1, heap);
                Item valueToRemove = heap[heap.Count - 1];
                heap.RemoveAt(heap.Count - 1);
                siftDown(0, heap.Count - 1, heap);
                return valueToRemove;
            }

            public void Insert(Item value)
            {
                heap.Add(value);
                siftUp(heap.Count - 1, heap);
            }

            public void swap(int i, int j, List<Item> heap)
            {
                Item temp = heap[j];
                heap[j] = heap[i];
                heap[i] = temp;
            }
        }

        //https://www.algoexpert.io/questions/count-inversions
        // O(nlogn) time | O(n) space - where n is the length of the array
        public int CountInversions(int[] array)
        {
            return countSubArrayInversions(array, 0, array.Length);
        }

        public int countSubArrayInversions(int[] array, int start, int end)
        {
            if (end - start <= 1)
            {
                return 0;
            }

            int middle = start + (end - start) / 2;
            int leftInversions = countSubArrayInversions(array, start, middle);
            int rightInversions = countSubArrayInversions(array, middle, end);
            int mergedArrayInversions =
              mergeSortAndCountInversions(array, start, middle, end);
            return leftInversions + rightInversions + mergedArrayInversions;
        }

        public int mergeSortAndCountInversions(
          int[] array, int start, int middle, int end
        )
        {
            List<int> sortedArray = new List<int>();
            int left = start;
            int right = middle;
            int inversions = 0;

            while (left < middle && right < end)
            {
                if (array[left] <= array[right])
                {
                    sortedArray.Add(array[left]);
                    left += 1;
                }
                else
                {
                    inversions += middle - left;
                    sortedArray.Add(array[right]);
                    right += 1;
                }
            }

            for (int idx = left; idx < middle; idx++)
            {
                sortedArray.Add(array[idx]);
            }

            for (int idx = right; idx < end; idx++)
            {
                sortedArray.Add(array[idx]);
            }

            for (int idx = 0; idx < sortedArray.Count; idx++)
            {
                int num = sortedArray[idx];
                array[start + idx] = num;
            }

            return inversions;
        }
        /*
      1. Two Sum
      https://leetcode.com/problems/two-sum/description
      */
        public int[] TwoSum(int[] nums, int target)
        {
            int[] result = new int[2];

            /*
                Approach 1: Brute Force
                Complexity Analysis
                •	Time complexity: O(n2).
                    For each element, we try to find its complement by looping through the rest of the array which takes O(n) time. Therefore, the time complexity is O(n2).
                •	Space complexity: O(1).
                    The space required does not depend on the size of the input array, so only constant space is used.
            */
            result = TwoSumNaive(nums, target);

            /*
Approach 2: Two-pass Hash Table
Complexity Analysis
•	Time complexity: O(n).
    We traverse the list containing n elements exactly twice. Since the hash table reduces the lookup time to O(1), the overall time complexity is O(n).
•	Space complexity: O(n).
    The extra space required depends on the number of items stored in the hash table, which stores exactly n elements.
*/
            result = TwoSumOptimal1(nums, target);

            /*
            Approach 3: One-pass Hash Table

            Complexity Analysis
            •	Time complexity: O(n).
                We traverse the list containing n elements only once. Each lookup in the table costs only O(1) time.
            •	Space complexity: O(n).
                The extra space required depends on the number of items stored in the hash table, which stores at most n elements.
            */
            result = TwoSumOptimal2(nums, target);

            return result;
        }

        private int[] TwoSumOptimal2(int[] nums, int target)
        {
            Dictionary<int, int> map = new Dictionary<int, int>();
            for (int i = 0; i < nums.Length; i++)
            {
                int complement = target - nums[i];
                if (map.ContainsKey(complement))
                {
                    return new int[] { map[complement], i };
                }

                map[nums[i]] = i;
            }

            return null;
        }

        private int[] TwoSumOptimal1(int[] nums, int target)
        {
            Dictionary<int, int> map = new Dictionary<int, int>();
            for (int i = 0; i < nums.Length; i++)
            {
                map[nums[i]] = i;
            }

            for (int i = 0; i < nums.Length; i++)
            {
                int complement = target - nums[i];
                if (map.ContainsKey(complement) && map[complement] != i)
                {
                    return new int[] { i, map[complement] };
                }
            }

            return null;
        }

        private int[] TwoSumNaive(int[] nums, int target)
        {
            for (int i = 0; i < nums.Length; i++)
            {
                for (int j = i + 1; j < nums.Length; j++)
                {
                    if (nums[j] == target - nums[i])
                    {
                        return new int[] { i, j };
                    }
                }
            }

            // In case there is no solution, return null
            return null;
        }
        /*
        474. Ones and Zeroes
        https://leetcode.com/problems/ones-and-zeroes/description/	
        */
        public int FindMaxForm(string[] strs, int m, int n)
        {
            /*
 Approach #1 Brute Force [Time Limit Exceeded]           
Complexity Analysis
•	Time complexity : O(2^l∗x).  2^l possible subsets, where l is the length of the list strs and x is the average string length.
•	Space complexity : O(1). Constant Space required.
            
            */
            int maxForm = FindMaxFormNaive(strs, m, n);
            /*
Approach #2 Better Brute Force [Time Limit Exceeded]

Complexity Analysis
•	Time complexity : O(2^l∗x). 2l possible subsets, where l is the length of the list strs and x is the average string length.
•	Space complexity : O(1). Constant Space required.
            
            */
            maxForm = FindMaxFormNaive2(strs, m, n);

            /*
Approach #3 Using Recursion [Time Limit Exceeded]
Complexity Analysis
•	Time complexity : O(2^l∗x). 2l possible subsets, where l is the length of the list strs and x is the average string length.
•	Space complexity : O(l). Depth of recursion tree grows upto l.
            
            */
            maxForm = FindMaxFormRec(strs, m, n);

            /*
Approach #4 Using Memoization [Accepted]
Complexity Analysis**
•	Time complexity : O(l∗m∗n). memo array of size l∗m∗n is filled, where l is the length of strs, m and n are the number of zeroes and ones respectively.
•	Space complexity : O(l∗m∗n). 3D array memo is used.           
            
            */
            maxForm = FindMaxFormMemo(strs, m, n);

            /*
Approach #5 Dynamic Programming 
Complexity Analysis
•	Time complexity : O(l∗m∗n). Three nested loops are their, where l is the length of strs, m and n are the number of zeroes and ones respectively.
•	Space complexity : O(m∗n). dp array of size m∗n is used
            
            */
            maxForm = FindMaxFormDP(strs, m, n);

            return maxForm;

        }

        public int FindMaxFormNaive(String[] strs, int m, int n)
        {
            int maxlen = 0;
            for (int i = 0; i < (1 << strs.Length); i++)
            {
                int zeroes = 0, ones = 0, len = 0;
                for (int j = 0; j < strs.Length; j++)
                {
                    if ((i & (1 << j)) != 0)
                    {
                        int[] count = CountZeroesOnes(strs[j]);
                        zeroes += count[0];
                        ones += count[1];
                        len++;
                    }
                }
                if (zeroes <= m && ones <= n)
                    maxlen = Math.Max(maxlen, len);
            }
            return maxlen;

        }
        public int[] CountZeroesOnes(String s)
        {
            int[] c = new int[2];
            for (int i = 0; i < s.Length; i++)
            {
                c[s[i] - '0']++;
            }
            return c;
        }

        public int FindMaxFormNaive2(String[] strs, int m, int n)
        {
            int maxlen = 0;
            for (int i = 0; i < (1 << strs.Length); i++)
            {
                int zeroes = 0, ones = 0, len = 0;
                for (int j = 0; j < 32; j++)
                {
                    if ((i & (1 << j)) != 0)
                    {
                        int[] count = CountZeroesOnes(strs[j]);
                        zeroes += count[0];
                        ones += count[1];
                        if (zeroes > m || ones > n)
                            break;
                        len++;
                    }
                }
                if (zeroes <= m && ones <= n)
                    maxlen = Math.Max(maxlen, len);
            }
            return maxlen;
        }
        public int FindMaxFormRec(String[] strs, int m, int n)
        {
            return Calculate(strs, 0, m, n);
        }
        public int Calculate(String[] strs, int i, int zeroes, int ones)
        {
            if (i == strs.Length)
                return 0;
            int[] count = CountZeroesOnes(strs[i]);
            int taken = -1;
            if (zeroes - count[0] >= 0 && ones - count[1] >= 0)
                taken = Calculate(strs, i + 1, zeroes - count[0], ones - count[1]) + 1;
            int not_taken = Calculate(strs, i + 1, zeroes, ones);
            return Math.Max(taken, not_taken);
        }
        public int FindMaxFormMemo(string[] strs, int m, int n)
        {
            int[][][] memo = new int[strs.Length][][];
            return Calculate(strs, 0, m, n, memo);
        }
        public int Calculate(string[] strs, int i, int zeroes, int ones, int[][][] memo)
        {
            if (i == strs.Length)
                return 0;
            if (memo[i][zeroes][ones] != 0)
                return memo[i][zeroes][ones];
            int[] count = CountZeroesOnes(strs[i]);
            int taken = -1;
            if (zeroes - count[0] >= 0 && ones - count[1] >= 0)
                taken = Calculate(strs, i + 1, zeroes - count[0], ones - count[1], memo) + 1;
            int not_taken = Calculate(strs, i + 1, zeroes, ones, memo);
            memo[i][zeroes][ones] = Math.Max(taken, not_taken);
            return memo[i][zeroes][ones];
        }

        public int FindMaxFormDP(String[] strs, int m, int n)
        {
            int[][] dp = new int[m + 1][];
            foreach (string s in strs)
            {
                int[] count = CountZeroesOnes(s);
                for (int zeroes = m; zeroes >= count[0]; zeroes--)
                    for (int ones = n; ones >= count[1]; ones--)
                        dp[zeroes][ones] = Math.Max(1 + dp[zeroes - count[0]][ones - count[1]], dp[zeroes][ones]);
            }
            return dp[m][n];
        }

        /*
14. Longest Common Prefix
https://leetcode.com/problems/longest-common-prefix/description/	

        */
        public string LongestCommonPrefix(string[] strs)
        {
            /*
Approach 1: Horizontal scanning (HS)
Complexity Analysis
•	Time complexity : O(S) , where S is the sum of all characters in all strings.
In the worst case all n strings are the same. The algorithm compares the string S1 with the other strings [S2…Sn] There are S character comparisons, where S is the sum of all characters in the input array.
•	Space complexity : O(1). We only used constant extra space


            */
            string longestCommonPrefix = LongestCommonPrefixHS(strs);
            /*
Approach 2: Vertical scanning (VS)           
  Complexity Analysis
•	Time complexity : O(S) , where S is the sum of all characters in all strings.
In the worst case there will be n equal strings with length m and the algorithm performs S=m⋅n character comparisons.
Even though the worst case is still the same as Approach 1, in the best case there are at most n⋅minLen comparisons where minLen is the length of the shortest string in the array.
•	Space complexity : O(1). We only used constant extra space
          
            */
            longestCommonPrefix = LongestCommonPrefixVS(strs);
            /*
Approach 3: Divide and conquer (DAC)           
Complexity Analysis
In the worst case we have n equal strings with length m
•	Time complexity : O(S), where S is the number of all characters in the array, S=m⋅n
Time complexity is 2⋅T(2n)+O(m). Therefore time complexity is O(S).
In the best case this algorithm performs O(minLen⋅n) comparisons, where minLen is the shortest string of the array
•	Space complexity : O(m⋅logn)
There is a memory overhead since we store recursive calls in the execution stack. There are logn recursive calls, each store need m space to store the result, so space complexity is O(m⋅logn)

            
            */
            longestCommonPrefix = LongestCommonPrefixDAC(strs);
            /*
 Approach 4: Binary search (BS)
Complexity Analysis
In the worst case we have n equal strings with length m
•	Time complexity : O(S⋅logm), where S is the sum of all characters in all strings.
The algorithm makes logm iterations, for each of them there are S=m⋅n comparisons, which gives in total O(S⋅logm) time complexity.
•	Space complexity : O(1). We only used constant extra space.


            */
            longestCommonPrefix = LongestCommonPrefixBS(strs);
            /*
Approach 5: Trie            
Complexity Analysis
In the worst case query q has length m and it is equal to all n strings of the array.
•	Time complexity : preprocessing O(S), where S is the number of all characters in the array, LCP query O(m).
Trie build has O(S) time complexity. To find the common prefix of q in the Trie takes in the worst case O(m).
•	Space complexity : O(S). We only used additional S extra space for the Trie.

            */
            longestCommonPrefix = LongestCommonPrefixTrie("q", strs);

            return longestCommonPrefix;
        }
        public string LongestCommonPrefixHS(string[] strs)
        {
            if (strs.Length == 0)
                return "";
            string prefix = strs[0];
            for (int i = 1; i < strs.Length; i++)
                while (strs[i].IndexOf(prefix) != 0)
                {
                    prefix = prefix.Substring(0, prefix.Length - 1);
                    if (prefix == "")
                        return "";
                }

            return prefix;
        }

        public string LongestCommonPrefixVS(string[] strs)
        {
            if (strs == null || strs.Length == 0)
                return "";
            for (int i = 0; i < strs[0].Length; i++)
            {
                char c = strs[0][i];
                for (int j = 1; j < strs.Length; j++)
                {
                    if (i == strs[j].Length || strs[j][i] != c)
                        return strs[0].Substring(0, i);
                }
            }

            return strs[0];
        }

        public string LongestCommonPrefixDAC(string[] strs)
        {
            if (strs == null || strs.Length == 0)
                return "";
            return LongestCommonPrefixDAC(strs, 0, strs.Length - 1);
        }

        private string LongestCommonPrefixDAC(string[] strs, int l, int r)
        {
            if (l == r)
            {
                return strs[l];
            }
            else
            {
                int mid = (l + r) / 2;
                var lcpLeft = LongestCommonPrefixDAC(strs, l, mid);
                var lcpRight = LongestCommonPrefixDAC(strs, mid + 1, r);
                return CommonPrefix(lcpLeft, lcpRight);
            }
        }

        private string CommonPrefix(string left, string right)
        {
            int min = Math.Min(left.Length, right.Length);
            for (int i = 0; i < min; i++)
            {
                if (left[i] != right[i])
                    return left.Substring(0, i);
            }

            return left.Substring(0, min);
        }
        public string LongestCommonPrefixBS(string[] strs)
        {
            if (strs == null || strs.Length == 0)
                return "";
            int minLen = Int32.MaxValue;
            foreach (string str in strs) minLen = Math.Min(minLen, str.Length);
            int low = 1;
            int high = minLen;
            while (low <= high)
            {
                int middle = (low + high) / 2;
                if (IsCommonPrefix(strs, middle))
                    low = middle + 1;
                else
                    high = middle - 1;
            }

            return strs[0].Substring(0, (low + high) / 2);
        }

        private bool IsCommonPrefix(string[] strs, int len)
        {
            string str1 = strs[0].Substring(0, len);
            for (int i = 1; i < strs.Length; i++)
                if (!strs[i].StartsWith(str1))
                    return false;
            return true;
        }
        public string LongestCommonPrefixTrie(string q, string[] strs)
        {
            if (strs == null || strs.Length == 0)
                return "";
            if (strs.Length == 1)
                return strs[0];
            Trie trie = new Trie();
            for (int i = 1; i < strs.Length; i++)
            {
                trie.Insert(strs[i]);
            }

            return trie.SearchLongestPrefix(q);
        }
        public class TrieNode
        {
            public TrieNode[] children = new TrieNode[26];
            public bool isEnd;
            private int linkCount;  // To count the number of children that are not null

            public TrieNode()
            {
                isEnd = false;
                for (int i = 0; i < 26; i++)
                {
                    children[i] = null;
                }

                linkCount = 0;
            }

            public void Put(char ch, TrieNode node)
            {
                int index = ch - 'a';
                if (children[index] == null)
                {
                    children[index] = node;
                    linkCount++;
                }
            }

            public bool ContainsKey(char ch)
            {
                return children[ch - 'a'] != null;
            }

            public int GetLinks()
            {
                return linkCount;
            }
        }

        public class Trie
        {
            private TrieNode root;

            public Trie()
            {
                root = new TrieNode();
            }

            public void Insert(string word)
            {
                TrieNode node = root;
                foreach (char c in word)
                {
                    if (!node.ContainsKey(c))
                    {
                        node.Put(c, new TrieNode());
                    }

                    node = node.children[c - 'a'];
                }

                node.isEnd = true;
            }

            public string SearchLongestPrefix(string word)
            {
                TrieNode node = root;
                StringBuilder prefix = new StringBuilder();
                foreach (char c in word)
                {
                    if (node.children[c - 'a'] != null && node.GetLinks() == 1 &&
                        !node.isEnd)
                    {
                        prefix.Append(c);
                        node = node.children[c - 'a'];
                    }
                    else
                    {
                        break;
                    }
                }

                return prefix.ToString();
            }
        }
        /*

        15. 3Sum
https://leetcode.com/problems/3sum/description/

        */
        public IList<IList<int>> ThreeSum(int[] nums)
        {

            /*

Approach 1: Two Pointers (TP)
Complexity Analysis
•	Time Complexity: O(n^2). twoSumII is O(n), and we call it n times.
Sorting the array takes O(nlogn), so overall complexity is O(nlogn+n^2). This is asymptotically equivalent to O(n^2).
•	Space Complexity: from O(logn) to O(n), depending on the implementation of the sorting algorithm. For the purpose of complexity analysis, we ignore the memory required for the output.


            */
            IList<IList<int>> tripletsSumToZero = ThreeSumTP(nums);
            /*
Approach 2: Hashset (HS)
Complexity Analysis	
Time Complexity: O(n^2). twoSum is O(n), and we call it n times.
Sorting the array takes O(nlogn), so overall complexity is O(nlogn+n^2). This is asymptotically equivalent to O(n^2).
•	Space Complexity: O(n) for the hashset.

            */
            tripletsSumToZero = ThreeSumHS(nums);
            /*
Approach 3: "No-Sort" (NS)
Complexity Analysis
•	Time Complexity: O(n^2). We have outer and inner loops, each going through n elements.
While the asymptotic complexity is the same, this algorithm is noticeably slower than the previous approach. Lookups in a hashset, though requiring a constant time, are expensive compared to the direct memory access.
•	Space Complexity: O(n) for the hashset/hashmap.
For the purpose of complexity analysis, we ignore the memory required for the output. However, in this approach we also store output in the hashset for deduplication. In the worst case, there could be O(n^2) triplets in the output, like for this example: [-k, -k + 1, ..., -1, 0, 1, ... k - 1, k]. Adding a new number to this sequence will produce n / 3 new triplets.

            */
            tripletsSumToZero = ThreeSumNS(nums);

            return tripletsSumToZero;

        }

        public IList<IList<int>> ThreeSumTP(int[] nums)
        {
            Array.Sort(nums);
            List<IList<int>> res = new List<IList<int>>();
            for (int i = 0; i < nums.Length && nums[i] <= 0; ++i)
                if (i == 0 || nums[i - 1] != nums[i])
                {
                    TwoSumII(nums, i, res);
                }

            return res;
        }

        void TwoSumII(int[] nums, int i, List<IList<int>> res)
        {
            int lo = i + 1, hi = nums.Length - 1;
            while (lo < hi)
            {
                int sum = nums[i] + nums[lo] + nums[hi];
                if (sum < 0)
                {
                    ++lo;
                }
                else if (sum > 0)
                {
                    --hi;
                }
                else
                {
                    res.Add(new List<int> { nums[i], nums[lo++], nums[hi--] });
                    while (lo < hi && nums[lo] == nums[lo - 1]) ++lo;
                }
            }
        }

        public IList<IList<int>> ThreeSumHS(int[] nums)
        {
            Array.Sort(nums);
            List<IList<int>> res = new List<IList<int>>();
            for (int i = 0; i < nums.Length && nums[i] <= 0; ++i)
                if (i == 0 || nums[i - 1] != nums[i])
                {
                    TwoSum(nums, i, res);
                }

            return res;
        }

        void TwoSum(int[] nums, int i, List<IList<int>> res)
        {
            HashSet<int> seen = new HashSet<int>();
            for (int j = i + 1; j < nums.Length; ++j)
            {
                int complement = -nums[i] - nums[j];
                if (seen.Contains(complement))
                {
                    res.Add(new List<int> { nums[i], nums[j], complement });
                    while (j + 1 < nums.Length && nums[j] == nums[j + 1]) ++j;
                }

                seen.Add(nums[j]);
            }
        }
        public IList<IList<int>> ThreeSumNS(int[] nums)
        {
            var res = new List<IList<int>>();
            Array.Sort(nums);
            for (int i = 0; i < nums.Length - 2; i++)
            {
                if (i == 0 || nums[i] != nums[i - 1])
                {
                    int lo = i + 1, hi = nums.Length - 1, sum = 0 - nums[i];
                    while (lo < hi)
                    {
                        if (nums[lo] + nums[hi] == sum)
                        {
                            res.Add(new List<int> { nums[i], nums[lo], nums[hi] });
                            while (lo < hi && nums[lo] == nums[lo + 1]) lo++;
                            while (lo < hi && nums[hi] == nums[hi - 1]) hi--;
                            lo++;
                            hi--;
                        }
                        else if (nums[lo] + nums[hi] < sum)
                            lo++;
                        else
                            hi--;
                    }
                }
            }

            return res;
        }

        /*
    167. Two Sum II - Input Array Is Sorted
    https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/description/	

    Approach 1: Two Pointers
    Complexity Analysis	
    •	Time complexity: O(n).
    The input array is traversed at most once. Thus the time complexity is O(n).
    •	Space complexity: O(1).
    We only use additional space to store two indices and the sum, so the space complexity is O(1).

        */
        public int[] TwoSumOfSortedArray(int[] numbers, int target)
        {
            int low = 0;
            int high = numbers.Length - 1;
            while (low < high)
            {
                int sum = numbers[low] + numbers[high];
                if (sum == target)
                {
                    return new int[] { low + 1, high + 1 };
                }
                else if (sum < target)
                {
                    ++low;
                }
                else
                {
                    --high;
                }
            }

            // In case there is no solution, return [-1, -1].
            return new int[] { -1, -1 };
        }

        /*
 16. 3Sum Closest			
https://leetcode.com/problems/3sum-closest/description/
       
        */
        public int ThreeSumClosest(int[] nums, int target)
        {
            /*
 Approach 1: Two Pointers (TP)           
Complexity Analysis
•	Time Complexity: O(n^2). We have outer and inner loops, each going through n elements.
Sorting the array takes O(nlogn), so overall complexity is O(nlogn+n^2). This is asymptotically equivalent to O(n^2).
•	Space Complexity: from O(logn) to O(n), depending on the implementation of the sorting algorithm.

            */
            int sumOfThreeNums = ThreeSumClosestTP(nums, target);

            /*
Approach 2: Binary Search (BS)
Complexity Analysis
•	Time Complexity: O(n^2logn). Binary search takes O(logn), and we do it n times in the inner loop. Since we are going through n elements in the outer loop, the overall complexity is O(n^2logn).
•	Space Complexity: from O(logn) to O(n), depending on the implementation of the sorting algorithm.
            
            */
            sumOfThreeNums = ThreeSumClosestBS(nums, target);

            return sumOfThreeNums;

        }

        public int ThreeSumClosestTP(int[] nums, int target)
        {
            int diff = Int32.MaxValue;
            int sz = nums.Length;
            Array.Sort(nums);
            for (int i = 0; i < sz && diff != 0; ++i)
            {
                int lo = i + 1;
                int hi = sz - 1;
                while (lo < hi)
                {
                    int sum = nums[i] + nums[lo] + nums[hi];
                    if (Math.Abs(target - sum) < Math.Abs(diff))
                    {
                        diff = target - sum;
                    }

                    if (sum < target)
                    {
                        ++lo;
                    }
                    else
                    {
                        --hi;
                    }
                }
            }

            return target - diff;
        }

        public int ThreeSumClosestBS(int[] nums, int target)
        {
            int diff = int.MaxValue, sz = nums.Length;
            Array.Sort(nums);
            for (int i = 0; i < sz && diff != 0; ++i)
            {
                for (int j = i + 1; j < sz - 1; ++j)
                {
                    int complement = target - nums[i] - nums[j];
                    int lo = j + 1, hi = sz;
                    while (lo < hi)
                    {
                        int mid = lo + (hi - lo) / 2;
                        if (nums[mid] <= complement)
                            lo = mid + 1;
                        else
                            hi = mid;
                    }

                    int hi_idx = lo, lo_idx = lo - 1;
                    if (hi_idx < sz &&
                        Math.Abs(complement - nums[hi_idx]) < Math.Abs(diff))
                        diff = complement - nums[hi_idx];
                    if (lo_idx > j &&
                        Math.Abs(complement - nums[lo_idx]) < Math.Abs(diff))
                        diff = complement - nums[lo_idx];
                }
            }

            return target - diff;
        }

        /*
   18. 4Sum
https://leetcode.com/problems/4sum/description/
     
        */
        public IList<IList<int>> FourSum(int[] nums, int target)
        {

            /*
    Approach 1: Two Pointers(TP)
    Complexity Analysis
    •	Time Complexity: O(n^(k−1)), or O(n^3) for 4Sum. We have k−2 loops, and twoSum is O(n).
    Note that for k>2, sorting the array does not change the overall time complexity.
    •	Space Complexity: O(n). We need O(k) space for the recursion. k can be the same as n in the worst case for the generalized algorithm.
    Note that, for the purpose of complexity analysis, we ignore the memory required for the output.

            */
            IList<IList<int>> uniqueQuadraplesToTarget = FourSumTP(nums, target);
            /*
      Approach 2: Hash (HS))
       Complexity Analysis
    •	Time Complexity: O(n^(k−1)), or O(n^3) for 4Sum. We have k−2 loops iterating over n elements, and twoSum is O(n).
    Note that for k>2, sorting the array does not change the overall time complexity.
    •	Space Complexity: O(n) for the hash set. The space needed for the recursion will not exceed O(n).

            */
            uniqueQuadraplesToTarget = FourSumHS(nums, target);

            return uniqueQuadraplesToTarget;

        }
        public IList<IList<int>> FourSumTP(int[] nums, int target)
        {
            Array.Sort(nums);
            return KSum(nums, target, 0, 4);
        }

        public IList<IList<int>> KSum(int[] nums, long target, int start, int k)
        {
            List<IList<int>> res = new List<IList<int>>();
            if (start == nums.Length)
            {
                return res;
            }

            long average_value = target / k;
            if (nums[start] > average_value ||
                average_value > nums[nums.Length - 1])
            {
                return res;
            }

            if (k == 2)
            {
                return TwoSum(nums, target, start);
            }

            for (int i = start; i < nums.Length; i++)
            {
                if (i == start || nums[i - 1] != nums[i])
                {
                    foreach (var subset in KSum(nums, target - nums[i], i + 1,
                                                k - 1))
                    {
                        var list = new List<int> { nums[i] };
                        list.AddRange(subset);
                        res.Add(list);
                    }
                }
            }

            return res;
        }

        public IList<IList<int>> TwoSum(int[] nums, long target, int start)
        {
            List<IList<int>> res = new List<IList<int>>();
            int lo = start, hi = nums.Length - 1;
            while (lo < hi)
            {
                int curr_sum = nums[lo] + nums[hi];
                if (curr_sum < target || (lo > start && nums[lo] == nums[lo - 1]))
                {
                    ++lo;
                }
                else if (curr_sum > target ||
                           (hi < nums.Length - 1 && nums[hi] == nums[hi + 1]))
                {
                    --hi;
                }
                else
                {
                    res.Add(new List<int> { nums[lo++], nums[hi--] });
                }
            }

            return res;
        }

        public IList<IList<int>> FourSumHS(int[] nums, int target)
        {
            Array.Sort(nums);
            return KSumHS(nums, target, 0, 4);
        }

        private IList<IList<int>> KSumHS(int[] nums, long target, int start, int k)
        {
            List<IList<int>> res = new List<IList<int>>();
            if (start == nums.Length)
            {
                return res;
            }

            long averageValue = target / k;
            if (nums[start] > averageValue ||
                averageValue > nums[nums.Length - 1])
            {
                return res;
            }

            if (k == 2)
            {
                return TwoSumHS(nums, target, start);
            }

            for (int i = start; i < nums.Length; ++i)
            {
                if (i == start || nums[i - 1] != nums[i])
                {
                    foreach (List<int> subset in KSumHS(nums, target - nums[i], i + 1,
                                                      k - 1))
                    {
                        List<int> temp = new List<int> { nums[i] };
                        temp.AddRange(subset);
                        res.Add(temp);
                    }
                }
            }

            return res;
        }

        public IList<IList<int>> TwoSumHS(int[] nums, long target, int start)
        {
            List<IList<int>> res = new List<IList<int>>();
            HashSet<long> s = new HashSet<long>();
            for (int i = start; i < nums.Length; ++i)
            {
                if (res.Count == 0 || res[res.Count - 1][1] != nums[i])
                {
                    if (s.Contains(target - nums[i]))
                    {
                        res.Add(new List<int> { (int)target - nums[i], nums[i] });
                    }
                }

                s.Add(nums[i]);
            }

            return res;
        }

        /*
454. 4Sum II		
https://leetcode.com/problems/4sum-ii/description/


        */
        public int FourSumCount(int[] nums1, int[] nums2, int[] nums3, int[] nums4)
        {

            /*
Approach 1: Hashmap/Dict
Complexity Analysis
•	Time Complexity: O(n^2). We have 2 nested loops to count sums, and another 2 nested loops to find complements.
•	Space Complexity: O(n^2) for the hashmap. There could be up to O(n^2) distinct a + b keys.

*/
            int numOfQuadruplesSumToZero = FourSumCountDict(nums1, nums2, nums3, nums4);

            /*
      Approach 2: kSum II
      Complexity Analysis
    •	Time Complexity: O(n^⌈k/2⌉), or O(n^2) for 4Sum II. We have to enumerate over at most n^⌊k/2⌋ sums in the left group and n^⌈k/2⌉ sums in the right group. Finally, we just need to check O*n^⌊k/2⌋ sums in the left group and search if their negated number exists in the right group.
    •	Space Complexity: O(n^⌈k/2⌉), similarly, we create a HashMap for each group to store all sums, which contains at most n^⌈k/2⌉ keys.

            */
            numOfQuadruplesSumToZero = FourSumCountKSumII(nums1, nums2, nums3, nums4);

            return numOfQuadruplesSumToZero;

        }
        public int FourSumCountDict(int[] A, int[] B, int[] C, int[] D)
        {
            int cnt = 0;
            Dictionary<int, int> dict = new Dictionary<int, int>();
            foreach (int a in A)
            {
                foreach (int b in B)
                {
                    dict.Add(a + b, dict.GetValueOrDefault(a + b, 0) + 1);
                }
            }
            foreach (int c in C)
            {
                foreach (int d in D)
                {
                    cnt += dict.GetValueOrDefault(-(c + d), 0);
                }
            }
            return cnt;
        }
        private int[][] lsts;

        public int FourSumCountKSumII(int[] A, int[] B, int[] C, int[] D)
        {
            lsts = new int[][] { A, B, C, D };
            int k = lsts.Length;
            Dictionary<int, int> left = sumCount(0, k / 2);
            Dictionary<int, int> right = sumCount(k / 2, k);
            int res = 0;
            foreach (int s in left.Keys)
                res += left[s] * right.GetValueOrDefault(-s, 0);
            return res;
        }

        private Dictionary<int, int> sumCount(int start, int end)
        {
            Dictionary<int, int> cnt = new Dictionary<int, int>();
            cnt.Add(0, 1);
            for (int i = start; i < end; i++)
            {
                Dictionary<int, int> map = new Dictionary<int, int>();
                foreach (int a in lsts[i])
                {
                    foreach (int total in cnt.Keys)
                    {
                        map.Add(total + a, map.GetValueOrDefault(total + a, 0) + cnt[total]);
                    }
                }
                cnt = map;
            }
            return cnt;
        }

        /*
        26. Remove Duplicates from Sorted Array
        https://leetcode.com/problems/remove-duplicates-from-sorted-array/description/

        Complexity Analysis
    Let N be the size of the input array.
    •	Time Complexity: O(N), since we only have 2 pointers, and both the pointers will traverse the array at most once.
    •	Space Complexity: O(1), since we are not using any extra space


        */
        public int RemoveDuplicates(int[] nums)
        {
            int insertIndex = 1;
            for (int i = 1; i < nums.Length; i++)
            {
                // We skip to next index if we see a duplicate element
                if (nums[i - 1] != nums[i])
                {
                    /* Storing the unique element at insertIndex index and
                       incrementing the insertIndex by 1 */
                    nums[insertIndex] = nums[i];
                    insertIndex++;
                }
            }

            return insertIndex;
        }

        /*
        80. Remove Duplicates from Sorted Array II
        https://leetcode.com/problems/remove-duplicates-from-sorted-array-ii/description/

        */
        public class RemoveDuplicatesSortedArrayIISolution
        {
            /*
            Approach 1: Popping Unwanted Duplicates (PUD)
            Complexity Analysis
•	Time Complexity: Let's see what the costly operations in our array are:
o	We have to iterate over all the elements in the array. Suppose that the original array contains N elements, the time taken here would be O(N).
o	Next, for every unwanted duplicate element, we will have to perform a delete operation and deletions in arrays are also O(N).
o	The worst case would be when all the elements in the array are the same. In that case, we would be performing N−2 deletions thus giving us O(N2) complexity for deletions
o	Overall complexity = O(N)+O(N^2)≡O(N^2).
•	Space Complexity: O(1) since we are modifying the array in-place.

            */
            public int RemoveDuplicatesSortedArrayIIPUD(int[] nums)
            {
                if (nums.Length == 0)
                {
                    return 0;
                }

                int i = 1; // Pointer for current index in the array
                int count = 1; // Count of the current element occurrences

                for (int j = 1; j < nums.Length; j++)
                {
                    if (nums[j] == nums[j - 1])
                    {
                        count++; // Increment count for the current element
                    }
                    else
                    {
                        count = 1; // Reset count for new element
                    }

                    if (count <= 2)
                    {
                        nums[i++] = nums[j]; // Update the array in place
                    }
                }

                return i; // Return the new array length
            }
            /*
            Approach 2: Overwriting unwanted duplicates (OUD)

            Complexity Analysis
Time Complexity: O(N) since we process each element exactly once.
Space Complexity: O(1).
            
            */
            public int RemoveDuplicatesSortedArrayIIOUD(int[] nums)
            {
                if (nums.Length == 0)
                {
                    return 0;
                }

                int i = 1; // Pointer to iterate through the array
                int j = 1; // Pointer to track position for valid elements
                int count = 1; // Count of occurrences of the current element

                while (i < nums.Length)
                {
                    if (nums[i] == nums[i - 1])
                    {
                        count++;
                        if (count > 2)
                        {
                            i++;
                            continue;
                        }
                    }
                    else
                    {
                        count = 1;
                    }
                    nums[j] = nums[i];
                    j++;
                    i++;
                }

                // Java arrays can't be resized like C++ vectors,
                // so we return the size directly.
                return j;
            }


        }

        /*
 27. Remove Element
https://leetcode.com/problems/remove-element/description/	
       
        */
        public int RemoveElement(int[] nums, int val)
        {
            /*
 
Approach 1: Two Pointers(TP)
 Complexity analysis
•	Time complexity : O(n).
Assume the array has a total of n elements, both i and j traverse at most 2n steps.
•	Space complexity : O(1).
          
            
            */
            int removedElement = RemoveElementTP(nums, val);
            /*
  Approach 2: Two Pointers - when elements to remove are rare          
Complexity analysis
•	Time complexity : O(n).
Both i and n traverse at most n steps. In this approach, the number of assignment operations is equal to the number of elements to remove. So it is more efficient if elements to remove are rare.
•	Space complexity : O(1).

            */
            removedElement = RemoveElementTPOptimal(nums, val);

            return removedElement;

        }
        public int RemoveElementTP(int[] nums, int val)
        {
            int i = 0;
            for (int j = 0; j < nums.Length; j++)
            {
                if (nums[j] != val)
                {
                    nums[i] = nums[j];
                    i++;
                }
            }

            return i;
        }
        public int RemoveElementTPOptimal(int[] nums, int val)
        {
            int i = 0;
            int n = nums.Length;
            while (i < n)
            {
                if (nums[i] == val)
                {
                    nums[i] = nums[n - 1];
                    // reduce array size by one
                    n--;
                }
                else
                {
                    i++;
                }
            }

            return n;
        }

        /*
        33. Search in Rotated Sorted Array
        https://leetcode.com/problems/search-in-rotated-sorted-array/description/

        */
        public int SearchInRotatedArray(int[] nums, int target)
        {

            /*
    Approach 1: Find Pivot Index + Binary Search (PIBS)        
    Complexity Analysis
Let n be the length of nums.
•	Time complexity: O(logn)
o	The algorithm requires one binary search to locate pivot, and at most 2 binary searches to find target. Each binary search takes O(logn) time.
•	Space complexity: O(1)
o	We only need to update several parameters left, right and mid, which takes O(1) space.
        
            */
            int indexOfTarget = SearchInRotatedSortedArrayPIBS(nums, target);
            /*
   Approach 2: Find Pivot Index + Binary Search with Shift (PIBSS)         
   Complexity Analysis
Let n be the length of nums.
•	Time complexity: O(logn)
o	The algorithm requires one binary search to locate pivot and one binary search over the shifted indices to find target. Each binary search takes O(logn) time.
•	Space complexity: O(1)
o	We only need to update several parameters left, right mid and shift, which takes O(1) space.
         
            */
            indexOfTarget = SearchInRotatedSortedArrayPIBSS(nums, target);
            /*
  Approach 3: One Binary Search (OBS)          
Complexity Analysis
Let n be the length of nums.
•	Time complexity: O(logn)
o	This algorithm only requires one binary search over nums.
•	Space complexity: O(1)
o	We only need to update several parameters left, right and mid, which takes O(1) space.

            */
            indexOfTarget = SearchInRotatedSortedArrayOBS(nums, target);

            return indexOfTarget;

        }
        public int SearchInRotatedSortedArrayPIBS(int[] nums, int target)
        {
            int n = nums.Length;
            int left = 0, right = n - 1;
            // Find the index of the pivot element (the smallest element)
            while (left <= right)
            {
                int mid = (left + right) / 2;
                if (nums[mid] > nums[n - 1])
                {
                    left = mid + 1;
                }
                else
                {
                    right = mid - 1;
                }
            }

            // Binary search over elements on the pivot element's left
            int answer = BinarySearch(nums, 0, left - 1, target);
            if (answer != -1)
            {
                return answer;
            }

            // Binary search over elements on the pivot element's right
            return BinarySearch(nums, left, n - 1, target);
        }

        // Binary search over an inclusive range [left_boundary ~ right_boundary]
        private int BinarySearch(int[] nums, int left_boundary, int right_boundary,
                                 int target)
        {
            int left = left_boundary, right = right_boundary;
            while (left <= right)
            {
                int mid = (left + right) / 2;
                if (nums[mid] == target)
                {
                    return mid;
                }
                else if (nums[mid] > target)
                {
                    right = mid - 1;
                }
                else
                {
                    left = mid + 1;
                }
            }

            return -1;
        }

        public int SearchInRotatedSortedArrayPIBSS(int[] nums, int target)
        {
            int n = nums.Length;
            int left = 0, right = n - 1;
            // Find the index of the pivot element (the smallest element)
            while (left <= right)
            {
                int mid = (left + right) / 2;
                if (nums[mid] > nums[n - 1])
                {
                    left = mid + 1;
                }
                else
                {
                    right = mid - 1;
                }
            }

            return ShiftedBinarySearch(nums, left, target);
        }

        // Shift elements in a circular manner, with the pivot element at index 0.
        // Then perform a regular binary search
        private int ShiftedBinarySearch(int[] nums, int pivot, int target)
        {
            int n = nums.Length;
            int shift = n - pivot;
            int left = (pivot + shift) % n;
            int right = (pivot - 1 + shift) % n;
            while (left <= right)
            {
                int mid = (left + right) / 2;
                if (nums[(mid - shift + n) % n] == target)
                {
                    return (mid - shift + n) % n;
                }
                else if (nums[(mid - shift + n) % n] > target)
                {
                    right = mid - 1;
                }
                else
                {
                    left = mid + 1;
                }
            }

            return -1;
        }
        public int SearchInRotatedSortedArrayOBS(int[] nums, int target)
        {
            int n = nums.Length;
            int left = 0, right = n - 1;
            while (left <= right)
            {
                int mid = left + (right - left) / 2;
                // Case 1: find target
                if (nums[mid] == target)
                {
                    return mid;
                }
                // Case 2: subarray on mid's left is sorted
                else if (nums[mid] >= nums[left])
                {
                    if (target >= nums[left] && target < nums[mid])
                    {
                        right = mid - 1;
                    }
                    else
                    {
                        left = mid + 1;
                    }
                }
                // Case 3: subarray on mid's right is sorted
                else
                {
                    if (target <= nums[right] && target > nums[mid])
                    {
                        left = mid + 1;
                    }
                    else
                    {
                        right = mid - 1;
                    }
                }
            }

            return -1;
        }
        /*
81. Search in Rotated Sorted Array II
https://leetcode.com/problems/search-in-rotated-sorted-array-ii/description/

  Approach 1: Binary Search      
Complexity Analysis
•	Time complexity : O(N) worst case, O(logN) best case, where N is the length of the input array.
Worst case: This happens when all the elements are the same and we search for some different element. At each step, we will only be able to reduce the search space by 1 since arr[mid] equals arr[start] and it's not possible to decide the relative position of target from arr[mid].
Example: [1, 1, 1, 1, 1, 1, 1], target = 2.
Best case: This happens when all the elements are distinct. At each step, we will be able to divide our search space into half just like a normal binary search.
This also answers the following follow-up question:
1.	Would this (having duplicate elements) affect the run-time complexity? How and why?
As we can see, by having duplicate elements in the array, we often miss the opportunity to apply binary search in certain search spaces. Hence, we get O(N) worst case (with duplicates) vs O(logN) best case complexity (without duplicates).
•	Space complexity : O(1)
        */
        public bool SearchInRotatedSortedArrayII(int[] nums, int target)
        {
            if (nums.Length == null)
                return false;
            int end = nums.Length - 1;
            int start = 0;
            while (start <= end)
            {
                int mid = start + (end - start) / 2;
                if (nums[mid] == target)
                    return true;
                if (nums[start] == nums[mid] && nums[end] == nums[mid])
                {
                    start++;
                    end--;
                }
                else if (nums[start] <= nums[mid])
                {
                    if (nums[start] <= target && target < nums[mid])
                        end = mid - 1;
                    else
                        start = mid + 1;
                }
                else
                {
                    if (nums[mid] < target && target <= nums[end])
                        start = mid + 1;
                    else
                        end = mid - 1;
                }
            }

            return false;
        }

        /*
        153. Find Minimum in Rotated Sorted Array
    https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/description/

    Complexity Analysis
    •	Time Complexity : Same as Binary Search O(logN)
    •	Space Complexity : O(1)

        */
        public int FindMinInRotatedSortedArray(int[] nums)
        {
            // If the list has just one element then return that element.
            if (nums.Length == 1)
            {
                return nums[0];
            }

            // Initializing left and right pointers.
            int left = 0, right = nums.Length - 1;

            // If the last element is greater than the first element then there is no rotation.
            // E.g. 1 < 2 < 3 < 4 < 5 < 7. Already sorted array.
            // Hence the smallest element is first element. A[0]
            if (nums[right] > nums[0])
            {
                return nums[0];
            }

            // Binary search way
            while (right >= left)
            {
                // Find the mid element
                int mid = left + (right - left) / 2;

                // If the mid element is greater than its next element then mid+1 element is the smallest
                // This point would be the point of change. From higher to lower value.
                if (nums[mid] > nums[mid + 1])
                {
                    return nums[mid + 1];
                }

                // If the mid element is lesser than its previous element then mid element is the smallest
                if (nums[mid - 1] > nums[mid])
                {
                    return nums[mid];
                }

                // If the mid elements value is greater than the 0th element this means
                // the least value is still somewhere to the right as we are still dealing with elements greater than nums[0]
                if (nums[mid] > nums[0])
                {
                    left = mid + 1;
                }
                else
                {
                    // If nums[0] is greater than the mid value then this means the smallest value is somewhere to the left
                    right = mid - 1;
                }
            }

            return Int32.MaxValue;
        }

        /*
        34. Find First and Last Position of Element in Sorted Array
        https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/description/

        Approach: Binary Search
        Complexity Analysis
        •	Time Complexity: O(logN) considering there are N elements in the array. This is because binary search takes logarithmic time to scan an array of N elements. Why? Because at each step we discard half of the array we are scanning and hence, we're done after a logarithmic number of steps. We simply perform binary search twice in this case.
        •	Space Complexity: O(1) since we only use space for a few variables and our result array, all of which require constant space.

        */
        public int[] FindFirstAndLastPosOfTargetInSortedArray(int[] nums, int target)
        {
            int firstOccurrence = this.FindBound(nums, target, true);
            if (firstOccurrence == -1)
            {
                return new int[] { -1, -1 };
            }

            int lastOccurrence = this.FindBound(nums, target, false);
            return new int[] { firstOccurrence, lastOccurrence };

        }
        private int FindBound(int[] nums, int target, bool isFirst)
        {
            int N = nums.Length;
            int begin = 0, end = N - 1;
            while (begin <= end)
            {
                int mid = (begin + end) / 2;
                if (nums[mid] == target)
                {
                    if (isFirst)
                    {
                        if (mid == begin || nums[mid - 1] != target)
                        {
                            return mid;
                        }

                        end = mid - 1;
                    }
                    else
                    {
                        if (mid == end || nums[mid + 1] != target)
                        {
                            return mid;
                        }

                        begin = mid + 1;
                    }
                }
                else if (nums[mid] > target)
                {
                    end = mid - 1;
                }
                else
                {
                    begin = mid + 1;
                }
            }

            return -1;
        }
        /*
35. Search Insert Position
https://leetcode.com/problems/search-insert-position/description/

Approach 1: Binary Search.
Complexity Analysis
•	Time complexity : O(logN).
•	Space complexity: O(1)

        */
        public int SearchInsertSortedArray(int[] nums, int target)
        {
            int pivot, left = 0, right = nums.Length - 1;
            while (left <= right)
            {
                pivot = left + (right - left) / 2;
                if (nums[pivot] == target)
                    return pivot;
                if (target < nums[pivot])
                    right = pivot - 1;
                else
                    left = pivot + 1;
            }

            return left;

        }

        /*
        41. First Missing Positive
        https://leetcode.com/problems/first-missing-positive/description/

        */
        public int FirstMissingPositive(int[] nums)
        {
            /*
Approach 1: Boolean Array (BA)
Complexity Analysis
Let n be the length of nums.
•	Time complexity: O(n)
Marking the values from nums in seen takes O(n).
We check for values 1 to n in seen, which takes O(n).
The total time complexity will be O(2n), which we can simplify to O(n).
•	Space complexity: O(n)
We initialize the array seen, which is size n + 1, so the space complexity is O(n).

            
            */
            int firstMissingPositive = FirstMissingPositiveBA(nums);
            /*
Approach 2: Index as a Hash Key (IHK)
 Complexity Analysis
Let n be the length of nums,
•	Time complexity: O(n)
We traverse nums using a for loop three separate times, so the time complexity is O(n).
•	Space complexity: O(n)
We modify the array nums and use it to determine the answer, so the space complexity is O(n).
nums is the input array, so the auxiliary space used is O(1).	
           
            
            */
            firstMissingPositive = FirstMissingPositiveIHK(nums);
            /*
            
  Approach 3: Cycle Sort  (CS)        
  Complexity Analysis
Let n be the length of nums.
•	Time complexity: O(n)
We loop through the elements in nums once, swapping elements to sort the array. Swapping takes constant time. Sorting nums using cycle sort takes O(n) time.
Iterating through the sorted array and finding the first missing positive can take up to O(n).
The total time complexity is O(2n), which simplifies to O(n).
•	Space complexity: O(n)
We modify the array nums and use it to determine the answer, so the space complexity is O(n).
nums is the input array, so the auxiliary space used is O(1).
          
            */
            firstMissingPositive = FirstMissingPositiveCS(nums);

            return firstMissingPositive;


        }
        public int FirstMissingPositiveBA(int[] nums)
        {
            int n = nums.Length;
            bool[] seen = new bool[n + 1];  // Array for lookup
                                            // Mark the elements from nums in the lookup array
            foreach (int num in nums)
            {
                if (num > 0 && num <= n)
                {
                    seen[num] = true;
                }
            }

            // Iterate through integers 1 to n
            // return smallest missing positive integer
            for (int i = 1; i <= n; i++)
            {
                if (!seen[i])
                {
                    return i;
                }
            }

            // If seen contains all elements 1 to n
            // the smallest missing positive number is n + 1
            return n + 1;
        }
        public int FirstMissingPositiveIHK(int[] nums)
        {
            int arrayLength = nums.Length;
            bool containsOne = false;

            // Replace negative numbers, zeros,
            // and numbers larger than n with 1s.
            // After this nums contains only positive numbers.
            for (int index = 0; index < arrayLength; index++)
            {
                // Check whether 1 is in the original array
                if (nums[index] == 1)
                {
                    containsOne = true;
                }
                if (nums[index] <= 0 || nums[index] > arrayLength)
                {
                    nums[index] = 1;
                }
            }

            if (!containsOne) return 1;

            // Mark whether integers 1 to n are in nums
            // Use index as a hash key and negative sign as a presence detector.
            for (int index = 0; index < arrayLength; index++)
            {
                int value = Math.Abs(nums[index]);
                if (value == arrayLength)
                {
                    nums[0] = -Math.Abs(nums[0]);
                }
                else
                {
                    nums[value] = -Math.Abs(nums[value]);
                }
            }

            // First positive in nums is smallest missing positive integer
            for (int index = 1; index < arrayLength; index++)
            {
                if (nums[index] > 0) return index;
            }

            // nums[0] stores whether n is in nums
            if (nums[0] > 0)
            {
                return arrayLength;
            }

            // If nums contains all elements 1 to n
            // the smallest missing positive number is n + 1
            return arrayLength + 1;
        }
        public int FirstMissingPositiveCS(int[] nums)
        {
            int n = nums.Length;
            // Use cycle sort to place positive elements smaller than n
            // at the correct index
            int i = 0;
            while (i < n)
            {
                int correctIdx = nums[i] - 1;
                if (nums[i] > 0 && nums[i] <= n && nums[i] != nums[correctIdx])
                {
                    Swap(nums, i, correctIdx);
                }
                else
                {
                    i++;
                }
            }

            // Iterate through nums
            // return smallest missing positive integer
            for (i = 0; i < n; i++)
            {
                if (nums[i] != i + 1)
                {
                    return i + 1;
                }
            }

            // If all elements are at the correct index
            // the smallest missing positive number is n + 1
            return n + 1;
        }

        /*
        53. Maximum Subarray
        https://leetcode.com/problems/maximum-subarray/description/

        */
        public int MaxSubArray(int[] nums)
        {
            /*
       Approach 1: Optimized Brute Force     
         Complexity Analysis
•	Time complexity: O(N^2), where N is the length of nums.
We use 2 nested for loops, with each loop iterating through nums.
•	Space complexity: O(1)
No matter how big the input is, we are only ever using 2 variables: ans and currentSubarray.
   
            */
            int maxSuAArray = MaxSubArrayNaiveOptimal(nums);
            /*
Approach 2: Dynamic Programming, Kadane's Algorithm (DPsKA)
Complexity Analysis
•	Time complexity: O(N), where N is the length of nums.
We iterate through every element of nums exactly once.
•	Space complexity: O(1)
No matter how long the input is, we are only ever using 2 variables: currentSubarray and maxSubarray.
          
            */
            maxSuAArray = MaxSubArrayDPKA(nums);
            /*
Approach 3: Divide and Conquer (Advanced) (DAC)
   Complexity Analysis
•	Time complexity: O(N⋅logN), where N is the length of nums.
On our first call to findBestSubarray, we use for loops to visit every element of nums. Then, we split the array in half and call findBestSubarray with each half. Both those calls will then iterate through every element in that half, which combined is every element of nums again. Then, both those halves will be split in half, and 4 more calls to findBestSubarray will happen, each with a quarter of nums. As you can see, every time the array is split, we still need to handle every element of the original input nums. We have to do this logN times since that's how many times an array can be split in half.
•	Space complexity: O(logN), where N is the length of nums.
The extra space we use relative to input size is solely occupied by the recursion stack. Each time the array gets split in half, another call of findBestSubarray will be added to the recursion stack, until calls start to get resolved by the base case - remember, the base case happens at an empty array, which occurs after logN calls.
         
            */
            maxSuAArray = MaxSubArrayDAC(nums);
            return maxSuAArray;

        }
        public int MaxSubArrayNaiveOptimal(int[] nums)
        {
            int max_subarray = int.MinValue;
            for (int i = 0; i < nums.Length; i++)
            {
                int current_subarray = 0;
                for (int j = i; j < nums.Length; j++)
                {
                    current_subarray += nums[j];
                    max_subarray = Math.Max(max_subarray, current_subarray);
                }
            }

            return max_subarray;
        }

        public int MaxSubArrayDPKA(int[] nums)
        {
            // Initialize our variables using the first element.
            int currentSubarray = nums[0];
            int maxSubarray = nums[0];
            // Start with the 2nd element since we already used the first one.
            for (int i = 1; i < nums.Length; i++)
            {
                // If current_subarray is negative, throw it away. Otherwise, keep
                // adding to it.
                currentSubarray = Math.Max(nums[i], currentSubarray + nums[i]);
                maxSubarray = Math.Max(maxSubarray, currentSubarray);
            }

            return maxSubarray;

        }

        private int[] numsArray;

        public int MaxSubArrayDAC(int[] nums)
        {
            numsArray = nums;
            // Our helper function is designed to solve this problem for
            // any array - so just call it using the entire input!
            return FindBestSubarray(0, numsArray.Length - 1);
        }

        private int FindBestSubarray(int left, int right)
        {
            // Base case - empty array.
            if (left > right)
            {
                return Int32.MinValue;
            }

            int mid = (left + right) / 2;
            int curr = 0;
            int bestLeftSum = 0;
            int bestRightSum = 0;
            // Iterate from the middle to the beginning.
            for (int i = mid - 1; i >= left; i--)
            {
                curr += numsArray[i];
                bestLeftSum = Math.Max(bestLeftSum, curr);
            }

            // Reset curr and iterate from the middle to the end.
            curr = 0;
            for (int i = mid + 1; i <= right; i++)
            {
                curr += numsArray[i];
                bestRightSum = Math.Max(bestRightSum, curr);
            }

            // The bestCombinedSum uses the middle element and the best possible sum
            // from each half.
            int bestCombinedSum = numsArray[mid] + bestLeftSum + bestRightSum;
            // Find the best subarray possible from both halves.
            int leftHalf = FindBestSubarray(left, mid - 1);
            int rightHalf = FindBestSubarray(mid + 1, right);
            // The largest of the 3 is the answer for any given input array.
            return Math.Max(bestCombinedSum, Math.Max(leftHalf, rightHalf));
        }

        /*
        189. Rotate Array
        https://leetcode.com/problems/rotate-array/description/

        */
        public class RotateArraySol
        {
            public void RotateArray(int[] nums, int k)
            {
                /*
         Approach 1: Brute Force       
          Complexity Analysis
    •	Time complexity: O(n×k).
    All the numbers are shifted by one step(O(n))
    k times.
    •	Space complexity: O(1). No extra space is used.

                */
                RotateArrayNaive(nums, k);
                /*
    Approach 2: Using Extra Array (EA)
    Complexity Analysis
    •	Time complexity: O(n).
    One pass is used to put the numbers in the new array.
    And another pass to copy the new array to the original one.
    •	Space complexity: O(n). Another array of the same size is used.

                */
                RotateArrayEA(nums, k);
                /*

    Approach 3: Using Cyclic Replacements (CR)
    Complexity Analysis
    •	Time complexity: O(n). Only one pass is used.
    •	Space complexity: O(1). Constant extra space is used.


                */
                RotateArrayCR(nums, k);

                /*

        Approach 4: Using Reverse (Rev)
        Complexity Analysis
        •	Time complexity: O(n). n elements are reversed a total of three times.
        •	Space complexity: O(1). No extra space is used.

                */
                RotateArrayRev(nums, k);

            }

            public void RotateArrayNaive(int[] nums, int k)
            {
                // speed up the rotation
                k %= nums.Length;
                int temp, previous;
                for (int i = 0; i < k; i++)
                {
                    previous = nums[nums.Length - 1];
                    for (int j = 0; j < nums.Length; j++)
                    {
                        temp = nums[j];
                        nums[j] = previous;
                        previous = temp;
                    }
                }
            }
            public void RotateArrayEA(int[] nums, int k)
            {
                int[] a = new int[nums.Length];
                for (int i = 0; i < nums.Length; i++)
                {
                    a[(i + k) % nums.Length] = nums[i];
                }
                for (int i = 0; i < nums.Length; i++)
                {
                    nums[i] = a[i];
                }
            }
            public void RotateArrayCR(int[] nums, int k)
            {
                k = k % nums.Length;
                int count = 0;
                for (int start = 0; count < nums.Length; start++)
                {
                    int current = start;
                    int prev = nums[start];
                    do
                    {
                        int next = (current + k) % nums.Length;
                        int temp = nums[next];
                        nums[next] = prev;
                        prev = temp;
                        current = next;
                        count++;
                    } while (start != current);
                }
            }
            public void RotateArrayRev(int[] nums, int k)
            {
                k %= nums.Length;
                Reverse(nums, 0, nums.Length - 1);
                Reverse(nums, 0, k - 1);
                Reverse(nums, k, nums.Length - 1);
            }

            public void Reverse(int[] nums, int start, int end)
            {
                while (start < end)
                {
                    int temp = nums[start];
                    nums[start] = nums[end];
                    nums[end] = temp;
                    start++;
                    end--;
                }
            }

        }



        /*
        75. Sort Colors
https://leetcode.com/problems/sort-colors/description/

Approach 1: One Pass

Complexity Analysis
•	Time complexity : O(N) since it's one pass along the array of length N.
•	Space complexity : O(1) since it's a constant space solution.

        */
        /*
   Dutch National Flag problem solution.
   */
        public void SortColors(int[] nums)
        {
            // For all idx < i : nums[idx < i] = 0
            // j is an index of elements under consideration
            var p0 = 0;
            var curr = 0;
            // For all idx > k : nums[idx > k] = 2
            var p2 = nums.Length - 1;
            while (curr <= p2)
            {
                if (nums[curr] == 0)
                {
                    // Swap p0-th and curr-th elements
                    // i++ and j++
                    int temp = nums[p0];
                    nums[p0++] = nums[curr];
                    nums[curr++] = temp;
                }
                else if (nums[curr] == 2)
                {
                    // Swap k-th and curr-th elements
                    // p2--
                    int temp = nums[curr];
                    nums[curr] = nums[p2];
                    nums[p2--] = temp;
                }
                else
                    curr++;
            }
        }

        /*
        280. Wiggle Sort
        https://leetcode.com/problems/wiggle-sort/description/

        */
        public class WiggleSortSolution
        {
            /*
            Approach 1: Sorting
       Complexity Analysis
Let, n be the size of nums.
•	Time complexity: O(n⋅log(n))
o	The time it takes to sort nums is O(n⋅log(n)).
o	We iterate over all the odd indices in O(n) time, then use the swap method to swap every odd index element with its next adjacent element in O(1) time per swap operation.
•	Space complexity: O(1)
o	For sorting, it depends on which algorithm we use to determine the space. However, sorting algorithms like heapsort take O(1) space.
o	Other than a few integers i, j, and temp, we do not need any space.
     
            */
            public void Swap(int[] nums, int i, int j)
            {
                int temp = nums[i];
                nums[i] = nums[j];
                nums[j] = temp;
            }

            public void WiggleSort(int[] nums)
            {
                Array.Sort(nums);
                for (int i = 1; i < nums.Length - 1; i += 2)
                {
                    Swap(nums, i, i + 1);
                }
            }
        }

        /*
        Approach 2: Greedy
Complexity Analysis
Let, n be the size of nums.
•	Time complexity: O(n)
o	We iterate over each nums element in O(n) time and, if necessary, use the swap method to swap the current element with the next element in O(1) time per swap operation.
•	Space complexity: O(1)
o	Other than a few integers i, j, and temp, we do not need any space.

        */

        public void WiggleSortGreedy(int[] nums)
        {
            for (int i = 0; i < nums.Length - 1; i++)
            {
                // Check if the current element is out of order with the next element
                // For even indices, the current element should be less than or equal to the next element
                // For odd indices, the current element should be greater than or equal to the next element
                if (((i % 2 == 0) && nums[i] > nums[i + 1])
                        || ((i % 2 == 1) && nums[i] < nums[i + 1]))
                {
                    Swap(nums, i, i + 1);
                }
            }
        }

        /*
        88. Merge Sorted Array
https://leetcode.com/problems/merge-sorted-array/description/

        */
        public class MergeSortedArraySol
        {

            /*
            Approach 1: Merge and sort

            Implementation
•	Time complexity: O((n+m)log(n+m)).
The cost of sorting a list of length x using a built-in sorting algorithm is O(xlogx). Because in this case, we're sorting a list of length m+n, we get a total time complexity of O((n+m)log(n+m)).
•	Space complexity: O(n), but it can vary.
Most programming languages have a built-in sorting algorithm that uses O(n) space.

            */
            public void MergeSortedArraySort(int[] nums1, int m, int[] nums2, int n)
            {
                for (int i = 0; i < n; i++)
                {
                    nums1[i + m] = nums2[i];
                }

                Array.Sort(nums1);
            }

            /*
            
Approach 2: Three Pointers (Start From the Beginning) (TPFrmBegin)
Complexity Analysis
•	Time complexity: O(n+m).
We are performing n+2⋅m reads and n+2⋅m writes. Because constants are ignored in Big O notation, this gives us a time complexity of O(n+m).
•	Space complexity: O(m).
We are allocating an additional array of length m


            */

            public void MergeSortedArrayTPFrmBegin(int[] nums1, int m, int[] nums2, int n)
            {
                // Make a copy of the first m elements of nums1.
                int[] nums1Copy = new int[m];
                Array.Copy(nums1, 0, nums1Copy, 0, m);
                // Read pointers for nums1Copy and nums2 respectively.
                int p1 = 0;
                int p2 = 0;
                // Compare elements from nums1Copy and nums2 and write the smallest to
                // nums1.
                for (int p = 0; p < m + n; p++)
                {
                    // We also need to ensure that p1 and p2 aren't over the boundaries
                    // of their respective arrays.
                    if (p2 >= n || (p1 < m && nums1Copy[p1] < nums2[p2]))
                    {
                        nums1[p] = nums1Copy[p1++];
                    }
                    else
                    {
                        nums1[p] = nums2[p2++];
                    }
                }
            }


            /*
            Approach 3: Three Pointers (Start From the End) (TPFrmEnd)

            Complexity Analysis
    •	Time complexity: O(n+m).
    Same as Approach 2.
    •	Space complexity: O(1).
    Unlike Approach 2, we're not using an extra array.


            */
            public void MergeSortedArrayTPFrmEnd(int[] nums1, int m, int[] nums2, int n)
            {
                // Set p1 and p2 to point to the end of their respective arrays.
                int p1 = m - 1;
                int p2 = n - 1;
                // And move p backward through the array, each time writing
                // the smallest value pointed at by p1 or p2.
                for (int p = m + n - 1; p >= 0; p--)
                {
                    if (p2 < 0)
                    {
                        break;
                    }

                    if (p1 >= 0 && nums1[p1] > nums2[p2])
                    {
                        nums1[p] = nums1[p1--];
                    }
                    else
                    {
                        nums1[p] = nums2[p2--];
                    }
                }
            }


        }


        /*
        2613. Beautiful Pairs
        https://leetcode.com/problems/beautiful-pairs/description/
        */
        public class BeautifulPairSol
        {
            private List<int[]> points = new List<int[]>();

            /*
            Approach: Divide And Conquer
            Time and Space Complexity
The given code is a Python class method solving a problem similar in nature to the closest pair of points problem, but with the added goal of finding the "beautiful" pair with the smallest indices. The beautifulPair function implements a divide-and-conquer strategy with additional sorting and filtering. It is important to note that the problem has been slightly modified to find a pair within two lists of integers with additional adherence to the index-based constraints.
Time Complexity:
•	The initial sorting of points has a time complexity of O(N log N), where N is the length of the list nums1 (and nums2, which is assumed to be the same length).
•	In the first loop to populate the points list and find duplicates, each iteration has constant time complexity, O(1), so the entire loop has a time complexity of O(N).
•	The dfs function represents a modified version of the divide-and-conquer algortihm to find the closest pair of points, with the recursive calls happening twice for each level of recursion, and an additional step to filter and sort t :
o	The recursive division of the dataset occurs log N times since the dataset is halved each time.
o	The filtering of points (within the dist threshold) and sorting of subarray t will have a worst-case complexity of O(N log N).
o	The nested loop comparison within the local sorted subset of points is O(N) in the worst case, as the inner loop breaks once d1 is exceeded. However, due to geometry, it is often seen as O(1) on average.
•	The overall time complexity of the dfs function is thus O(N log^2 N) due to the combination of recursive splitting (logarithmic) and sorting within each recursive call (logarithmic).
So, the total time complexity, taken by the sum of all contributing factors, is O(N log N) + O(N) + O(N log^2 N), which simplifies to O(N log^2 N) for large N.
Space Complexity:
•	Intermediate lists such as t in the dfs function can potentially store all points, leading to a space complexity of O(N).
•	The points list which stores the input points with their indices incurs a space complexity of O(N).
•	The recursion stack of the dfs function will use O(log N) space due to the divide-and-conquer approach.
The total space complexity is therefore the sum of these, dominated by the space required for storage of point information, resulting in O(N) space complexity.

            */
            // Main method to find the 'beautiful pair' according to the problem statement
            public int[] DivideAndConquer(int[] nums1, int[] nums2)
            {
                int length = nums1.Length;
                Dictionary<long, List<int>> pairsList = new Dictionary<long, List<int>>();
                for (int i = 0; i < length; ++i)
                {
                    long compositeNum = ComputeCompositeNumber(nums1[i], nums2[i]);
                    // Map the composite number to its indices in the arrays
                    if (!pairsList.ContainsKey(compositeNum))
                        pairsList[compositeNum] = new List<int>();
                    pairsList[compositeNum].Add(i);
                }
                for (int i = 0; i < length; ++i)
                {
                    long compositeNum = ComputeCompositeNumber(nums1[i], nums2[i]);
                    // Quick check for a beautiful pair if more than one occurrence is found
                    if (pairsList[compositeNum].Count > 1)
                    {
                        return new int[] { i, pairsList[compositeNum][1] };
                    }
                    // Store points along with their original indexes for later processing
                    points.Add(new int[] { nums1[i], nums2[i], i });
                }
                // Sort points based on the value of the first element
                points.Sort((a, b) => a[0].CompareTo(b[0]));
                // Recursively find the beautiful pair by using divide-and-conquer strategy
                int[] answer = FindClosestPair(0, points.Count - 1);
                // Return the indices of the pair found
                return new int[] { answer[1], answer[2] };
            }

            // Helper method to create a composite number for easy mapping
            private long ComputeCompositeNumber(int x, int y)
            {
                return x * 100000L + y;
            }

            // Method to calculate Manhattan distance between two points
            private int Distance(int x1, int y1, int x2, int y2)
            {
                return Math.Abs(x1 - x2) + Math.Abs(y1 - y2);
            }

            // Helper method that implements the divide-and-conquer approach to find the closest pair
            private int[] FindClosestPair(int left, int right)
            {
                if (left >= right)
                {
                    // Return maximum possible value if no pair found
                    return new int[] { int.MaxValue, -1, -1 };
                }
                int middle = (left + right) >> 1; // Find the midpoint
                int pivotX = points[middle][0];
                // Recursive calls to find the smallest pair in each half
                int[] resultLeft = FindClosestPair(left, middle);
                int[] resultRight = FindClosestPair(middle + 1, right);
                // Determine the smaller distance of the pairs found
                if (resultLeft[0] > resultRight[0]
                    || (resultLeft[0] == resultRight[0] && (resultLeft[1] > resultRight[1]
                    || (resultLeft[1] == resultRight[1] && resultLeft[2] > resultRight[2]))))
                {
                    resultLeft = resultRight;
                }
                List<int[]> filteredPoints = new List<int[]>();
                for (int i = left; i <= right; ++i)
                {
                    // Filtering points that can possibly form the closest pair
                    if (Math.Abs(points[i][0] - pivotX) <= resultLeft[0])
                    {
                        filteredPoints.Add(points[i]);
                    }
                }
                // Sort the filtered points based on the second dimension
                filteredPoints.Sort((a, b) => a[1].CompareTo(b[1]));
                for (int i = 0; i < filteredPoints.Count; ++i)
                {
                    for (int j = i + 1; j < filteredPoints.Count; ++j)
                    {
                        // No farther points need to be checked after a certain threshold
                        if (filteredPoints[j][1] - filteredPoints[i][1] > resultLeft[0])
                        {
                            break;
                        }
                        int firstIndex = Math.Min(filteredPoints[i][2], filteredPoints[j][2]);
                        int secondIndex = Math.Max(filteredPoints[i][2], filteredPoints[j][2]);
                        // Calculate the Manhattan distance between the pair of points
                        int d = Distance(filteredPoints[i][0], filteredPoints[i][1],
                                         filteredPoints[j][0], filteredPoints[j][1]);
                        // Update the result if a closer pair is found
                        if (d < resultLeft[0] || (d == resultLeft[0] && (firstIndex < resultLeft[1]
                            || (firstIndex == resultLeft[1] && secondIndex < resultLeft[2]))))
                        {
                            resultLeft = new int[] { d, firstIndex, secondIndex };
                        }
                    }
                }
                return resultLeft;
            }
        }

        /* 2366. Minimum Replacements to Sort the Array
        https://leetcode.com/problems/minimum-replacements-to-sort-the-array/description/
         */
        class MinimumReplacementToSortArraySol
        {
            /*
            Approach: Greedy
            Complexity Analysis
Let n be the size of nums.
•	Time complexity: O(n)
o	We iterate over nums once in reverse.
o	At each step, we calculate num_elements, answer and nums[i], which takes O(1) time.
•	Space complexity: O(1)
o	We're modifying nums in place and not using any additional data structures that scale with the size of the input.
o	Note that some interviewers might not want you to modify the input as it is not considered good practice in real-world coding. If that's the case, you could slightly modify the algorithm to use an integer to track the most recently split numbers.

            */
            public long WithGreedy(int[] nums)
            {
                long answer = 0;
                int n = nums.Length;

                // Start from the second last element, as the last one is always sorted.
                for (int i = n - 2; i >= 0; i--)
                {
                    // No need to break if they are already in order.
                    if (nums[i] <= nums[i + 1])
                    {
                        continue;
                    }

                    // Count how many elements are made from breaking nums[i].
                    long numElements = (long)(nums[i] + nums[i + 1] - 1) / (long)nums[i + 1];

                    // It requires numElements - 1 replacement operations.
                    answer += numElements - 1;

                    // Maximize nums[i] after replacement.
                    nums[i] = nums[i] / (int)numElements;
                }

                return answer;
            }
        }

        /* 1526. Minimum Number of Increments on Subarrays to Form a Target Array
        https://leetcode.com/problems/minimum-number-of-increments-on-subarrays-to-form-a-target-array/description/
         */
        public class MinNumberOperationsSol
        {
            public int MinNumberOperations(int[] target)
            {
                int count = target[0];

                for (int i = 1; i < target.Length; i++)
                    count += Math.Max(target[i] - target[i - 1], 0);

                return count;

            }
        }


        /* 472. Concatenated Words
        https://leetcode.com/problems/concatenated-words/description/
         */
        public class FindAllConcatenatedWordsInADictionarySol
        {
            /*
            
Approach 1: Dynamic Programming
Complexity Analysis
Here, N is the total number of strings in the array words, namely words.Length, and M is the length of the longest string in the array words.
•	Time complexity: O(M^3⋅N).
Although we use HashSet, we need to consider the cost to calculate the hash value of a string internally which would be O(M). So putting all words into the HashSet takes O(N∗M).
For each word, the i and j loops take O(M^2). The internal logic to take the substring and search in the HashSet needs to calculate the hash value for the substring too, and it should take another O(M), so for each word, the time complexity is O(M^3) and the total time complexity for N words is O(M^3⋅N)
•	Space complexity: O(N⋅M).
This is just the space to save all words in the dictionary, if we don't take M as a constant.

            */
            public List<string> FindAllConcatenatedWordsInADictionary(string[] words)
            {
                HashSet<string> dictionary = new HashSet<string>(words);
                List<string> concatenatedWords = new List<string>();

                foreach (string word in words)
                {
                    int wordLength = word.Length;
                    bool[] dynamicProgramming = new bool[wordLength + 1];
                    dynamicProgramming[0] = true;

                    for (int i = 1; i <= wordLength; ++i)
                    {
                        for (int j = (i == wordLength ? 1 : 0); !dynamicProgramming[i] && j < i; ++j)
                        {
                            dynamicProgramming[i] = dynamicProgramming[j] && dictionary.Contains(word.Substring(j, i - j));
                        }
                    }

                    if (dynamicProgramming[wordLength])
                    {
                        concatenatedWords.Add(word);
                    }
                }
                return concatenatedWords;
            }

            /*
            Approach 2: DFS
Complexity Analysis
Here, N is the total number of strings in the array words, namely words.Length, and M is the length of the longest string in the array words.
•	Time complexity: O(M^3⋅N).
For each word, the constructed graph has M nodes and O(M^2) edges, and the DFS algorithm for reachability is O(M^2) without considering the time complexities of substring and HashSet. If we consider everything, the time complexity to check one word is O(M^3) and the total time complexity to check all words is O(M^3⋅N).
•	Space complexity: O(N⋅M).
This is the space to save all words in the dictionary, if we don't take M as a constant, there is also O(M) for the call stack to execute DFS, which wouldn't affect the space complexity anyways.

            */
            public List<string> DFS(string[] words)
            {
                HashSet<string> dictionary = new HashSet<string>(words);
                List<string> answer = new List<string>();
                foreach (string word in words)
                {
                    int length = word.Length;
                    bool[] visited = new bool[length];
                    if (DepthFirstSearch(word, 0, visited, dictionary))
                    {
                        answer.Add(word);
                    }
                }
                return answer;
            }
            private bool DepthFirstSearch(string word, int length, bool[] visited, HashSet<string> dictionary)
            {
                if (length == word.Length)
                {
                    return true;
                }
                if (visited[length])
                {
                    return false;
                }
                visited[length] = true;
                for (int i = word.Length - (length == 0 ? 1 : 0); i > length; --i)
                {
                    if (dictionary.Contains(word.Substring(length, i - length))
                        && DepthFirstSearch(word, i, visited, dictionary))
                    {
                        return true;
                    }
                }
                return false;
            }
        }



        /* 493. Reverse Pairs
        https://leetcode.com/problems/reverse-pairs/description/
         */

        public class ReversePairsSol
        {
            /*
            1. binary search tree (BST)-based solution


            */
            public int BST(int[] nums)
            {
                int res = 0;
                Node root = null;

                foreach (int ele in nums)
                {
                    res += Search(root, 2L * ele + 1);
                    root = Insert(root, ele);
                }

                return res;
                int Search(Node root, long val)
                {
                    if (root == null)
                    {
                        return 0;
                    }
                    else if (val == root.val)
                    {
                        return root.cnt;
                    }
                    else if (val < root.val)
                    {
                        return root.cnt + Search(root.left, val);
                    }
                    else
                    {
                        return Search(root.right, val);
                    }
                }
                Node Insert(Node root, int val)
                {
                    if (root == null)
                    {
                        root = new Node(val);
                    }
                    else if (val == root.val)
                    {
                        root.cnt++;
                    }
                    else if (val < root.val)
                    {
                        root.left = Insert(root.left, val);
                    }
                    else
                    {
                        root.cnt++;
                        root.right = Insert(root.right, val);
                    }

                    return root;
                }

            }


            public class Node
            {
                public int val, cnt;
                public Node left, right;

                public Node(int val)
                {
                    this.val = val;
                    this.cnt = 1;
                }
            }
            /*
2. binary indexed tree (BIT)-based solution

*/
            public int BIT(int[] nums)
            {
                int result = 0;
                int[] copy = new int[nums.Length];
                Array.Copy(nums, copy, nums.Length);
                int[] bit = new int[copy.Length + 1];

                Array.Sort(copy);

                foreach (int element in nums)
                {
                    result += Search(bit, Index(copy, 2L * element + 1));
                    Insert(bit, Index(copy, element));
                }

                return result;
                int Search(int[] bit, int index)
                {
                    int sum = 0;

                    while (index < bit.Length)
                    {
                        sum += bit[index];
                        index += index & -index;
                    }

                    return sum;
                }

                void Insert(int[] bit, int index)
                {
                    while (index > 0)
                    {
                        bit[index] += 1;
                        index -= index & -index;
                    }
                }
            }

            private int Index(int[] array, long value)
            {
                int left = 0, right = array.Length - 1, middle = 0;

                while (left <= right)
                {
                    middle = left + ((right - left) >> 1);

                    if (array[middle] >= value)
                    {
                        right = middle - 1;
                    }
                    else
                    {
                        left = middle + 1;
                    }
                }

                return left + 1;
            }

            /*
         3. MergeSort

         */
            public int MergeSort(int[] numbers)
            {
                return ReversePairsSub(numbers, 0, numbers.Length - 1);
            }

            private int ReversePairsSub(int[] numbers, int left, int right)
            {
                if (left >= right) return 0;

                int middle = left + ((right - left) >> 1);
                int result = ReversePairsSub(numbers, left, middle) + ReversePairsSub(numbers, middle + 1, right);

                int i = left, j = middle + 1, k = 0, p = middle + 1;
                int[] merge = new int[right - left + 1];

                while (i <= middle)
                {
                    while (p <= right && numbers[i] > 2 * numbers[p]) p++;
                    result += p - (middle + 1);

                    while (j <= right && numbers[i] >= numbers[j]) merge[k++] = numbers[j++];
                    merge[k++] = numbers[i++];
                }

                while (j <= right) merge[k++] = numbers[j++];

                System.Array.Copy(merge, 0, numbers, left, merge.Length);

                return result;
            }
        }


        /* 315. Count of Smaller Numbers After Self
        https://leetcode.com/problems/count-of-smaller-numbers-after-self/description/
         */
        class CountSmallerNumberAfterSelfSol
        {
            /*
            
Approach 1: Segment Tree
Complexity Analysis
Let N be the length of nums and M be the difference between the maximum and minimum values in nums.
Note that for convenience, we fix M=2∗10^4 in the above implementations.
	Time Complexity: O(Nlog(M)).
We need to iterate over nums. For each element, we spend O(log(M)) to find the number of smaller elements after it, and spend O(log(M)) time to update the counts. In total, we need O(N⋅log(M))=O(Nlog(M)) time.
	Space Complexity: O(M), since we need, at most, an array of size 2M+2 to store the segment tree.
We need at most M+1 buckets, where the extra 1 is for the value 0. For the segment tree, we need twice the number of buckets, which is (M+1)×2=2M+2.

            */
            public List<int> UsingSegmentTree(int[] nums)
            {
                int offset = 10000; // offset negative to non-negative
                int size = 2 * 10000 + 1; // total possible values in nums
                int[] segmentTree = new int[size * 2];
                List<int> result = new List<int>();

                for (int i = nums.Length - 1; i >= 0; i--)
                {
                    int smallerCount = Query(0, nums[i] + offset, segmentTree, size);
                    result.Add(smallerCount);
                    Update(nums[i] + offset, 1, segmentTree, size);
                }
                result.Reverse();
                return result;
            }

            // implement segment tree
            private void Update(int index, int value, int[] tree, int size)
            {
                index += size; // shift the index to the leaf
                               // update from leaf to root
                tree[index] += value;
                while (index > 1)
                {
                    index /= 2;
                    tree[index] = tree[index * 2] + tree[index * 2 + 1];
                }
            }

            private int Query(int left, int right, int[] tree, int size)
            {
                // return sum of [left, right)
                int result = 0;
                left += size; // shift the index to the leaf
                right += size;
                while (left < right)
                {
                    // if left is a right node
                    // bring the value and move to parent's right node
                    if (left % 2 == 1)
                    {
                        result += tree[left];
                        left++;
                    }
                    // else directly move to parent
                    left /= 2;
                    // if right is a right node
                    // bring the value of the left node and move to parent
                    if (right % 2 == 1)
                    {
                        right--;
                        result += tree[right];
                    }
                    // else directly move to parent
                    right /= 2;
                }
                return result;
            }
            /* Approach 2: Binary Indexed Tree (Fenwick Tree)
            Complexity Analysis
            Let N be the length of nums and M be the difference between the maximum and minimum values in nums.
            Note that for convenience, we fix M=2∗10^4 in the above implementations.
                Time Complexity: O(Nlog(M)).
            We need to iterate over nums. For each element, we spend O(log(M)) to find the number of smaller elements after it, and spend O(log(M)) time to update the counts. In total, we need O(N⋅log(M))=O(Nlog(M)) time.
                Space Complexity: O(M), since we need, at most, an array of size M+2 to store the BIT.
            We need at most M+1 buckets, where the extra 1 is for the value 0. The BIT requires an extra dummy node, so the size is (M+1)+1=M+2.

             */
            public List<int> UsingBIT(int[] nums)
            {
                int offset = 10000; // offset negative to non-negative
                int size = 2 * 10000 + 2; // total possible values in nums plus one dummy
                int[] tree = new int[size];
                List<int> result = new List<int>();

                for (int i = nums.Length - 1; i >= 0; i--)
                {
                    int smallerCount = Query(nums[i] + offset, tree);
                    result.Add(smallerCount);
                    Update(nums[i] + offset, 1, tree, size);
                }
                result.Reverse();
                return result;
                // implement Binary Index Tree
                void Update(int index, int value, int[] tree, int size)
                {
                    index++; // index in BIT is 1 more than the original index
                    while (index < size)
                    {
                        tree[index] += value;
                        index += index & -index;
                    }
                }
            }
            private int Query(int index, int[] tree)
            {
                // return sum of [0, index)
                int result = 0;
                while (index >= 1)
                {
                    result += tree[index];
                    index -= index & -index;
                }
                return result;
            }
            /* 
                        Approach 3: Merge Sort
                        Complexity Analysis
            Let N be the length of nums.
                Time Complexity: O(Nlog(N)). We need to perform a merge sort which takes O(Nlog(N)) time. All other operations take at most O(N) time.
                Space Complexity: O(N), since we need a constant number of arrays of size O(N).

                         */
            public List<int> UsingMergeSort(int[] nums)
            {
                int arrayLength = nums.Length;
                int[] resultArray = new int[arrayLength];
                int[] indicesArray = new int[arrayLength]; // record the index. we are going to sort this array
                for (int index = 0; index < arrayLength; index++)
                {
                    indicesArray[index] = index;
                }
                // sort indices with their corresponding values in nums, i.e., nums[indices[i]]
                MergeSort(indicesArray, 0, arrayLength, resultArray, nums);
                // change int[] to List to return
                List<int> resultListToReturn = new List<int>(arrayLength);
                foreach (int count in resultArray)
                {
                    resultListToReturn.Add(count);
                }
                return resultListToReturn;
            }

            private void MergeSort(int[] indicesArray, int leftIndex, int rightIndex, int[] resultArray, int[] nums)
            {
                if (rightIndex - leftIndex <= 1)
                {
                    return;
                }
                int midIndex = (leftIndex + rightIndex) / 2;
                MergeSort(indicesArray, leftIndex, midIndex, resultArray, nums);
                MergeSort(indicesArray, midIndex, rightIndex, resultArray, nums);
                Merge(indicesArray, leftIndex, rightIndex, midIndex, resultArray, nums);
            }

            private void Merge(int[] indicesArray, int leftIndex, int rightIndex, int midIndex, int[] resultArray, int[] nums)
            {
                // merge [leftIndex, midIndex) and [midIndex, rightIndex)
                int leftCurrentIndex = leftIndex; // current index for the left array
                int rightCurrentIndex = midIndex; // current index for the right array
                                                  // use temp to temporarily store sorted array
                List<int> tempList = new List<int>(rightIndex - leftIndex);
                while (leftCurrentIndex < midIndex && rightCurrentIndex < rightIndex)
                {
                    if (nums[indicesArray[leftCurrentIndex]] <= nums[indicesArray[rightCurrentIndex]])
                    {
                        // rightCurrentIndex - midIndex numbers jump to the left side of indicesArray[leftCurrentIndex]
                        resultArray[indicesArray[leftCurrentIndex]] += rightCurrentIndex - midIndex;
                        tempList.Add(indicesArray[leftCurrentIndex]);
                        leftCurrentIndex++;
                    }
                    else
                    {
                        tempList.Add(indicesArray[rightCurrentIndex]);
                        rightCurrentIndex++;
                    }
                }
                // when one of the subarrays is empty
                while (leftCurrentIndex < midIndex)
                {
                    // rightCurrentIndex - midIndex numbers jump to the left side of indicesArray[leftCurrentIndex]
                    resultArray[indicesArray[leftCurrentIndex]] += rightCurrentIndex - midIndex;
                    tempList.Add(indicesArray[leftCurrentIndex]);
                    leftCurrentIndex++;
                }
                while (rightCurrentIndex < rightIndex)
                {
                    tempList.Add(indicesArray[rightCurrentIndex]);
                    rightCurrentIndex++;
                }
                // restore from temp
                for (int k = leftIndex; k < rightIndex; k++)
                {
                    indicesArray[k] = tempList[k - leftIndex];
                }
            }

        }


        /* 
        327. Count of Range Sum
        https://leetcode.com/problems/count-of-range-sum/description/ */
        public class CountRangeSumSol
        {
            /*
            Approach 1:Divide and Conquer 
            */
            public int UsingDivideAndConquer(int[] nums, int lower, int upper)
            {

                if (nums == null || nums.Length == 0 || lower > upper) return 0;
                return CountRangeSumSub(nums, 0, nums.Length - 1, lower, upper);

            }
            private int CountRangeSumSub(int[] nums, int l, int r, int lower, int upper)
            {
                if (l == r) return nums[l] >= lower && nums[r] <= upper ? 1 : 0;  // base case

                int m = l + (r - l) / 2;
                long[] arr = new long[r - m];  // prefix array for the second subarray
                long sum = 0;
                int count = 0;

                for (int i = m + 1; i <= r; i++)
                {
                    sum += nums[i];
                    arr[i - (m + 1)] = sum; // compute the prefix array
                }

                Array.Sort(arr);  // sort the prefix array

                // Here we can compute the suffix array element by element.
                // For each element in the suffix array, we compute the corresponding
                // "insertion" indices of the modified bounds in the sorted prefix array
                // then the number of valid ranges sums will be given by the indices difference.
                // I modified the bounds to be "double" to avoid duplicate elements.
                sum = 0;
                for (int i = m; i >= l; i--)
                {
                    sum += nums[i];
                    count += FindIndex(arr, upper - sum + 0.5) - FindIndex(arr, lower - sum - 0.5);
                }

                return CountRangeSumSub(nums, l, m, lower, upper) + CountRangeSumSub(nums, m + 1, r, lower, upper) + count;
            }

            // binary search function
            private int FindIndex(long[] arr, double val)
            {
                int l = 0, r = arr.Length - 1, m = 0;

                while (l <= r)
                {
                    m = l + (r - l) / 2;

                    if (arr[m] <= val)
                    {
                        l = m + 1;
                    }
                    else
                    {
                        r = m - 1;
                    }
                }

                return l;
            }

            /*
          Approach 2: Prefix-Array Divide and Conquer 
          */
            public int PrefixArrayDivideAndConquer(int[] nums, int lower, int upper)
            {
                if (nums == null || nums.Length == 0 || lower > upper) return 0;

                long[] prefixArray = new long[nums.Length + 1];

                for (int i = 1; i < prefixArray.Length; i++)
                {
                    prefixArray[i] = prefixArray[i - 1] + nums[i - 1];
                }

                return CountRangeSum(prefixArray, 0, prefixArray.Length - 1, lower, upper);
            }

            private int CountRangeSum(long[] prefixArray, int l, int r, int lower, int upper)
            {
                if (l >= r) return 0;

                int m = l + (r - l) / 2;

                int count = CountRangeSum(prefixArray, l, m, lower, upper) + CountRangeSum(prefixArray, m + 1, r, lower, upper);

                long[] mergedArray = new long[r - l + 1];
                int i = l, j = m + 1, k = m + 1, p = 0, q = m + 1;

                while (i <= m)
                {
                    while (j <= r && prefixArray[j] - prefixArray[i] < lower) j++;
                    while (k <= r && prefixArray[k] - prefixArray[i] <= upper) k++;
                    count += k - j;

                    while (q <= r && prefixArray[q] < prefixArray[i]) mergedArray[p++] = prefixArray[q++];
                    mergedArray[p++] = prefixArray[i++];
                }

                while (q <= r) mergedArray[p++] = prefixArray[q++];

                Array.Copy(mergedArray, 0, prefixArray, l, mergedArray.Length);

                return count;
            }
            /*
            binary indexed tree or Fenwick Tree 
            */
            public int UsingBIT(int[] nums, int lower, int upper)
            {
                long[] prefixSum = new long[nums.Length + 1];
                long[] candidates = new long[3 * prefixSum.Length + 1];
                int index = 0;
                candidates[index++] = prefixSum[0];
                candidates[index++] = lower + prefixSum[0] - 1;
                candidates[index++] = upper + prefixSum[0];

                for (int i = 1; i < prefixSum.Length; i++)
                {
                    prefixSum[i] = prefixSum[i - 1] + nums[i - 1];
                    candidates[index++] = prefixSum[i];
                    candidates[index++] = lower + prefixSum[i] - 1;
                    candidates[index++] = upper + prefixSum[i];
                }

                candidates[index] = long.MinValue; // avoid getting root of the binary indexed tree when doing binary search
                Array.Sort(candidates);

                int[] binaryIndexedTree = new int[candidates.Length];

                // build up the binary indexed tree with only elements from the prefix array "prefixSum"
                for (int i = 0; i < prefixSum.Length; i++)
                {
                    AddValue(binaryIndexedTree, Array.BinarySearch(candidates, prefixSum[i]), 1);
                }

                int count = 0;

                for (int i = 0; i < prefixSum.Length; i++)
                {
                    // get rid of visited elements by adding -1 to the corresponding tree nodes
                    AddValue(binaryIndexedTree, Array.BinarySearch(candidates, prefixSum[i]), -1);

                    // add the total number of valid elements with upper bound (upper + prefixSum[i])
                    count += Query(binaryIndexedTree, Array.BinarySearch(candidates, upper + prefixSum[i]));

                    // minus the total number of valid elements with lower bound (lower + prefixSum[i] - 1)
                    count -= Query(binaryIndexedTree, Array.BinarySearch(candidates, lower + prefixSum[i] - 1));
                }

                return count;
            }

            private void AddValue(int[] binaryIndexedTree, int index, int value)
            {
                while (index < binaryIndexedTree.Length)
                {
                    binaryIndexedTree[index] += value;
                    index += index & -index;
                }
            }

            private int Query(int[] binaryIndexedTree, int index)
            {
                int sum = 0;

                while (index > 0)
                {
                    sum += binaryIndexedTree[index];
                    index -= index & -index;
                }

                return sum;
            }

        }



        /* 3116. Kth Smallest Amount With Single Denomination Combination
        https://leetcode.com/problems/kth-smallest-amount-with-single-denomination-combination/description/
         */

        public class FindKthSmallestWithSingleDenomCombiSol
        {
            /*
            Approach: Math | Inclusion-Exclusion Principle | Binary Search 
            Complexity
•	Time complexity: O(2^n)×log(k), where n = len(coins)
•	Space complexity: O(2^n)

            */
            public int MathWithBinarySearch(int[] coins, int k)
            {
                int n = coins.Length;
                Dictionary<int, List<long>> dic = new Dictionary<int, List<long>>();
                for (int i = 1; i <= n; i++)
                {
                    foreach (var comb in GetCombinations(coins, i))
                    {
                        long lcm = GetLCM(comb);
                        if (!dic.ContainsKey(comb.Length))
                        {
                            dic[comb.Length] = new List<long>();
                        }
                        dic[comb.Length].Add(lcm);
                    }
                }

                long Count(Dictionary<int, List<long>> dic, long target)
                {
                    long ans = 0;
                    for (int i = 1; i <= n; i++)
                    {
                        foreach (var lcm in dic[i])
                        {
                            ans += target / lcm * (long)Math.Pow(-1, i + 1);
                        }
                    }
                    return ans;
                }

                long start = coins.Min(), end = coins.Min() * k;
                while (start + 1 < end)
                {
                    long mid = (start + end) / 2;
                    if (Count(dic, mid) >= k)
                    {
                        end = mid;
                    }
                    else
                    {
                        start = mid;
                    }
                }
                if (Count(dic, start) >= k)
                {
                    return (int)start;
                }
                else
                {
                    return (int)end;
                }
            }

            private IEnumerable<int[]> GetCombinations(int[] coins, int k)
            {
                return (IEnumerable<int[]>)coins.Combinations(k); //TODO: Test this code
            }

            private long GetLCM(int[] nums)
            {
                long lcm = 1;
                foreach (var num in nums)
                {
                    lcm = LCM(lcm, num);
                }
                return lcm;
            }

            private long LCM(long a, long b)
            {
                return a * b / GCD(a, b);
            }

            private long GCD(long a, long b)
            {
                return b == 0 ? a : GCD(b, a % b);
            }

        }



        /* 2025. Maximum Number of Ways to Partition an Array
        https://leetcode.com/problems/maximum-number-of-ways-to-partition-an-array/description/
         */
        class WaysToPartitionSol
        {
            public int WaysToPartition(int[] nums, int k)
            {
                int length = nums.Length;
                long[] prefixSum = new long[length], suffixSum = new long[length];

                prefixSum[0] = nums[0];
                suffixSum[length - 1] = nums[length - 1];
                for (int index = 1; index < length; index++)
                {
                    prefixSum[index] = prefixSum[index - 1] + nums[index];
                    suffixSum[length - index - 1] = suffixSum[length - index] + nums[length - index - 1];
                }

                Dictionary<long, int> leftCount = new Dictionary<long, int>();
                Dictionary<long, int> rightCount = new Dictionary<long, int>();

                for (int index = 0; index < length - 1; index++)
                {
                    long difference = prefixSum[index] - suffixSum[index + 1];
                    if (!rightCount.ContainsKey(difference)) rightCount[difference] = 0;
                    rightCount[difference]++;
                }

                int resultCount = 0;
                if (rightCount.ContainsKey(0)) resultCount += rightCount[0];

                for (int index = 0; index < length; index++)
                {
                    long difference = k - nums[index];
                    int currentCount = 0;
                    if (rightCount.ContainsKey(-difference)) currentCount += rightCount[-difference];
                    if (leftCount.ContainsKey(difference)) currentCount += leftCount[difference];

                    resultCount = Math.Max(resultCount, currentCount);

                    if (index < length - 1)
                    {
                        difference = prefixSum[index] - suffixSum[index + 1];
                        if (!leftCount.ContainsKey(difference)) leftCount[difference] = 0;
                        leftCount[difference]++;
                        if (rightCount.ContainsKey(difference))
                        {
                            if (rightCount[difference] <= 1) rightCount.Remove(difference);
                            else rightCount[difference]--;
                        }
                    }
                }

                return resultCount;
            }
        }

        /* 3139. Minimum Cost to Equalize Array
        https://leetcode.com/problems/minimum-cost-to-equalize-array/description/
         */

        public class MinCostToEqualizeArraySol
        {
            public int MinCostToEqualizeArray(int[] array, int cost1, int cost2)
            {
                int maxElement = array[0], minElement = array[0], arrayLength = array.Length, modulo = 1000000007;
                long totalSum = 0;
                foreach (int element in array)
                {
                    minElement = Math.Min(minElement, element);
                    maxElement = Math.Max(maxElement, element);
                    totalSum += element;
                }
                totalSum = 1L * maxElement * arrayLength - totalSum;

                // case 1
                if (cost1 * 2 <= cost2 || arrayLength <= 2)
                {
                    return (int)((totalSum * cost1) % modulo);
                }

                // case 2
                long option1 = Math.Max(0L, (maxElement - minElement) * 2L - totalSum);
                long option2 = totalSum - option1;
                long result = (option1 + option2 % 2) * cost1 + option2 / 2 * cost2;

                // case 3
                totalSum += option1 / (arrayLength - 2) * arrayLength;
                option1 %= arrayLength - 2;
                option2 = totalSum - option1;
                result = Math.Min(result, (option1 + option2 % 2) * cost1 + option2 / 2 * cost2);

                // case 4
                for (int index = 0; index < 2; index++)
                {
                    totalSum += arrayLength;
                    result = Math.Min(result, totalSum % 2 * cost1 + totalSum / 2 * cost2);
                }
                return (int)(result % modulo);
            }
        }


        /* 2263. Make Array Non-decreasing or Non-increasing
        https://leetcode.com/problems/make-array-non-decreasing-or-non-increasing/description/
         */
        public class MakeArrayNonDecreaseOrNonIncreaseSol
        {
            /* Approach 1: DP O(n^2): */
            public int DP(int[] nums)
            {
                var levels = new HashSet<int>(nums);
                var sortedLevels = new List<int>(levels);
                sortedLevels.Sort();
                var nums2 = new int[nums.Length];
                for (int i = 0; i < nums.Length; i++)
                {
                    nums2[nums.Length - 1 - i] = nums[i];
                }
                return Math.Min(Helper(nums, sortedLevels.ToArray()), Helper(nums2, sortedLevels.ToArray()));

                int Helper(int[] nums, int[] levels)
                {
                    var dp = new Dictionary<int, int>();
                    foreach (var num in nums)
                    {
                        var currentResult = int.MaxValue;
                        foreach (var level in levels)
                        {
                            var previousResult = dp.ContainsKey(level) ? dp[level] : 0;
                            currentResult = Math.Min(currentResult, previousResult + Math.Abs(num - level));
                            dp[level] = currentResult;
                        }
                    }
                    return dp.ContainsKey(levels[levels.Length - 1]) ? dp[levels[levels.Length - 1]] : 0;
                }
            }

            /* Approach 2: Max Heap  O(nlg(n)): */
            public int MaxHeap(int[] nums)
            {
                var levels = nums.Distinct().OrderBy(x => x).ToArray();
                var reversedNums = new int[nums.Length];
                for (int i = 0; i < nums.Length; i++)
                {
                    reversedNums[nums.Length - 1 - i] = nums[i];
                }
                return Math.Min(Helper(nums, levels), Helper(reversedNums, levels));
                int Helper(int[] nums, int[] levels)
                {
                    var priorityQueue = new PriorityQueue<int, int>();
                    var result = 0;
                    foreach (var number in nums) //TODO: Fix below PQ code
                    {
                        if (priorityQueue.Count > 0 && number < (-priorityQueue.Peek()))
                        {
                            result += (-priorityQueue.Dequeue()) - number;
                            priorityQueue.Enqueue(number, -number);
                        }
                        priorityQueue.Enqueue(number, -number);
                    }
                    return result;
                }
            }


        }

        /* 2302. Count Subarrays With Score Less Than K
        https://leetcode.com/problems/count-subarrays-with-score-less-than-k/solutions/
         */
        public long CountSubarrays(int[] nums, long k)
        {
            /* 1. Subarrays + Less Than K = Slide Window 
            Complexity
Time O(n)
Space O(1)

            */
            long res = 0, cur = 0;
            for (int j = 0, i = 0; j < nums.Length; ++j)
            {
                cur += nums[j];
                while (cur * (j - i + 1) >= k)
                    cur -= nums[i++];
                res += j - i + 1;
            }
            return res;
        }


        /* 3134. Find the Median of the Uniqueness Array
        https://leetcode.com/problems/find-the-median-of-the-uniqueness-array/description/
         */
        public class MedianOfUniquenessArraySol
        {
            /* Approach 1: Binary Search + Sliding Window
            Complexity
Time O(nlogn)
Space O(n)

             */
            public int BinarySearchWithSlidingWindow(int[] array)
            {
                int arrayLength = array.Length, leftBoundary = 1, rightBoundary = arrayLength;
                long totalSum = (long)arrayLength * (arrayLength + 1) / 2;
                while (leftBoundary < rightBoundary)
                {
                    int midPoint = (leftBoundary + rightBoundary) / 2;
                    if (AtMost(array, midPoint) * 2 >= totalSum)
                    {
                        rightBoundary = midPoint;
                    }
                    else
                    {
                        leftBoundary = midPoint + 1;
                    }
                }
                return leftBoundary;
            }

            private long AtMost(int[] array, int k)
            {
                long result = 0;
                Dictionary<int, int> countMap = new Dictionary<int, int>();
                int startIndex = 0;
                for (int endIndex = 0; endIndex < array.Length; endIndex++)
                {
                    if (countMap.ContainsKey(array[endIndex]))
                    {
                        countMap[array[endIndex]]++;
                    }
                    else
                    {
                        countMap[array[endIndex]] = 1;
                    }
                    while (countMap.Count > k)
                    {
                        countMap[array[startIndex]]--;
                        if (countMap[array[startIndex]] == 0)
                        {
                            countMap.Remove(array[startIndex]);
                        }
                        startIndex++;
                    }
                    result += endIndex - startIndex + 1;
                }
                return result;
            }
        }


        /* 629. K Inverse Pairs Array
        https://leetcode.com/problems/k-inverse-pairs-array/description/
         */

        public class KInversePairsSol
        {
            /* Approach 1: Brute Force - Generate every permutation of array
            Complexity Analysis
•	Time complexity : O(n!⋅nlogn). A total of n! permutations will be generated. We need O(nlogn) time to find the number of inverse pairs in every such permutation, by making use of merge sort. Here, n refers to the given integer n.
•	Space complexity : O(n). Each array generated during the permutations will require n space.

             */

            /*              Approach 2: Using Recursion with Memoization
Complexity Analysis
•	Time complexity : O(nk min(n,k)). The function kInversePairs is called nk times to fill the memo array. Each function call itself takes O(min(n,k)) time at worse.
•	Space complexity : O(nk). memo array of constant size n⋅k is used. The depth of recursion tree can go upto n.

             */
            int[][] memo = new int[1001][];
            public int RecurWithMemo(int n, int k)
            {
                if (n == 0)
                    return 0;
                if (k == 0)
                    return 1;
                if (memo[n][k] != null)
                    return memo[n][k];
                int inv = 0;
                for (int i = 0; i <= Math.Min(k, n - 1); i++)
                    inv = (inv + RecurWithMemo(n - 1, k - i)) % 1000000007;
                memo[n][k] = inv;
                return inv;
            }

            /*             Approach 3: Dynamic Programming
Complexity Analysis
•	Time complexity : O(nk min(n,k)). The dp table of size n⋅k is filled once. Filling each entry takes O(min(n,k)) time.
•	Space complexity : O(n∗k). dp array of size n⋅k is used.

             */
            public int DP(int n, int k)
            {
                int[][] dp = new int[n + 1][];
                for (int i = 1; i <= n; i++)
                {
                    for (int j = 0; j <= k; j++)
                    {
                        if (j == 0)
                            dp[i][j] = 1;
                        else
                        {
                            for (int p = 0; p <= Math.Min(j, i - 1); p++)
                                dp[i][j] = (dp[i][j] + dp[i - 1][j - p]) % 1000000007;
                        }
                    }
                }
                return dp[n][k];
            }


            /* Approach 4: Dynamic Programming with Cumulative Sum
            Complexity Analysis
•	Time complexity : O(nk). dp array of size n⋅k is filled once.
•	Space complexity : O(nk). dp array of size n⋅k is used.
             */
            public int DPWithCumulativeSum(int n, int k)
            {
                int[][] dp = new int[n + 1][];
                int M = 1000000007;
                for (int i = 1; i <= n; i++)
                {
                    for (int j = 0; j <= k; j++)
                    {
                        if (j == 0)
                            dp[i][j] = 1;
                        else
                        {
                            int val = (dp[i - 1][j] + M - ((j - i) >= 0 ? dp[i - 1][j - i] : 0)) % M;
                            dp[i][j] = (dp[i][j - 1] + val) % M;
                        }
                    }
                }
                return ((dp[n][k] + M - (k > 0 ? dp[n][k - 1] : 0)) % M);
            }

            /*
            Approach 5: Another Optimized Dynamic Programming Approach
            Complexity Analysis
            •	Time complexity : O(nk). dp array of size (n+1)⋅(k+1) is filled once.
            •	Space complexity : O(nk). dp array of size (n+1)⋅(k+1) is used.

             */

            public int DPWithCumulativeSumOptimal(int n, int k)
            {
                int[][] dp = new int[n + 1][];
                int M = 1000000007;
                for (int i = 1; i <= n; i++)
                {
                    for (int j = 0; j <= k && j <= i * (i - 1) / 2; j++)
                    {
                        if (i == 1 && j == 0)
                        {
                            dp[i][j] = 1;
                            break;
                        }
                        else if (j == 0)
                            dp[i][j] = 1;
                        else
                        {
                            int val = (dp[i - 1][j] + M - ((j - i) >= 0 ? dp[i - 1][j - i] : 0)) % M;
                            dp[i][j] = (dp[i][j - 1] + val) % M;
                        }
                    }
                }
                return dp[n][k];
            }


            /* Approach 6: Once Again Memoization
            Complexity Analysis
            •	Time complexity : O(nk). nxk entries in the memo array are filled once.
            •	Space complexity : O(nk). memo array of size n⋅k is used.

             */
            int M = 1000000007;
            public int DPRecWithMemo(int n, int k)
            {
                return ((Recur(n, k) + M - (k > 0 ? Recur(n, k - 1) : 0)) % M);
            }
            private int Recur(int n, int k)
            {
                if (n == 0)
                    return 0;
                if (k == 0)
                    return 1;
                if (memo[n][k] != null)
                    return memo[n][k];
                int val = (Recur(n - 1, k) + M - ((k - n) >= 0 ? Recur(n - 1, k - n) : 0)) % M;
                memo[n][k] = (Recur(n, k - 1) + val) % M;
                return memo[n][k];
            }

            /*
            Approach 7: 1-D Dynamic Programmming
            Complexity Analysis
            •	Time complexity : O(n∗k). dp array of size k+1 is filled n+1 times.
            •	Space complexity : O(k). dp array of size (k+1) is used.

             */
            public int DPSpaceOptimal(int n, int k)
            {
                int[] dp = new int[k + 1];
                int M = 1000000007;
                for (int i = 1; i <= n; i++)
                {
                    int[] temp = new int[k + 1];
                    temp[0] = 1;
                    for (int j = 1; j <= k; j++)
                    {
                        int val = (dp[j] + M - ((j - i) >= 0 ? dp[j - i] : 0)) % M;
                        temp[j] = (temp[j - 1] + val) % M;
                    }
                    dp = temp;
                }
                return ((dp[k] + M - (k > 0 ? dp[k - 1] : 0)) % M);
            }

        }


        /* 1354. Construct Target Array With Multiple Sums
        https://leetcode.com/problems/construct-target-array-with-multiple-sums/description/
         */
        public class IsPossibleToConstructTargetArrayWithMultipleSumsSol
        {
            /*
                         Approach 1: Working Backward
                     Complexity Analysis
            Let n be the length of the target array. Let k be the maximum of the target array.
            •	Time Complexity : O(n+klogn).
            Making the initial heap is O(n).
            Then, each heap operation (add and remove) is O(logn).
            The k comes from the cost of reducing the largest item. In the worst case, it is massive. For example, consider a target of [1, 1_000_000_000]. One step of the algorithm will reduce it to [1, 999_999_999]. And the next is [1, 999_999_998]. You probably see where this is going.
            Because we don't know whether n or klogn is larger, we add them.
            While for this problem, we're told the maximum possible value in the array is 1,000,000,000, we don't consider these limits for big-oh notation. Because the time taken varies with the maximum number, we consider this time complexity to be pseudo-polynomial. For the [1, 1_000_000_000] case, it is unworkably slow on any typical computer.
            •	Space Complexity : O(n).
            The heap requires O(n) extra space. There are always n or n−1 items on the heap at any one time.
            O(1) space would be possible, by converting the target array directly into a heap. The Python code does this, although the line we used to make the target array negative is "technically" O(n) (some Python interpreters might delete and replace this with an O(1) operation behind the scenes). If space was a big issue (it isn't generally), then it's useful to know that these optimizations are at least possible.
            The fact that this algorithm is pseudo-polynomial, and has to cope with large n values, and extremely large k values is a big limitation. Luckily there are a few tweaks we can make to the algorithm that will make it polynomial.

             */

            public bool WorkingBackward(int[] target)
            {

                // Handle the n = 1 case.
                if (target.Length == 1)
                {
                    return target[0] == 1;
                }

                int totalSum = target.Sum();

                //Max Heap
                PriorityQueue<int, int> priorityQueue = new PriorityQueue<int, int>();
                foreach (int number in target)
                {
                    priorityQueue.Enqueue(number, -number); // Using negative to mimic max-heap
                }

                while (priorityQueue.Peek() > 1)
                {
                    int largest = priorityQueue.Dequeue();
                    int x = largest - (totalSum - largest);
                    if (x < 1) return false;
                    priorityQueue.Enqueue(x, -x);
                    totalSum = totalSum - largest + x;
                }

                return true;
            }
            /* 
       Approach 2: Working Backward with Optimizations 
Complexity Analysis
•	Time Complexity : O(n+logk⋅logn).
Creating a heap is O(n).
At each step, we were removing at least 41 of the total sum. The original total sum is 2⋅k, because k is the largest element, and we know that if the algorithm continues, then the rest can't add up to more than k. So, we need to take O(logk) steps to reduce the array down. Each of these steps is the cost of a heap add and remove, i.e. O(logn). In total, this is O(logk⋅logn).
•	Space Complexity : O(n).
Same as above.

       */

            public bool WorkingBackwardWithOptimizations(int[] target)
            {
                // Handle the n = 1 case.
                if (target.Length == 1)
                {
                    return target[0] == 1;
                }

                int totalSum = 0;
                foreach (int num in target)
                {
                    totalSum += num;
                }

                // Using default minheap as a max heap to simulate PriorityQueue with reverse order
                PriorityQueue<int, int> maxHeap = new PriorityQueue<int, int>();
                foreach (int num in target)
                {
                    maxHeap.Enqueue(num, -num); // Using negative to mimic max-heap
                }

                while (maxHeap.Peek() > 1)
                {
                    int largest = maxHeap.Dequeue();
                    int rest = totalSum - largest;

                    // This will only occur if n = 2.
                    if (rest == 1)
                    {
                        return true;
                    }
                    int x = largest % rest;

                    // If x is now 0 (invalid) or didn't
                    // change, then we know this is impossible.
                    if (x == 0 || x == largest)
                    {
                        return false;
                    }
                    maxHeap.Enqueue(x, -x);
                    totalSum = totalSum - largest + x;
                }

                return true;
            }
        }

        /* 
        220. Contains Duplicate III
        https://leetcode.com/problems/contains-duplicate-iii/description/
         */

        class ContainsNearbyAlmostDuplicateSol
        {
            /* Approach #1 (Naive Linear Search) [Time Limit Exceeded]
            Complexity Analysis
•	Time complexity: O(n min(k,n)).
It costs O(min(k,n)) time for each linear search. Note that we do at most n comparisons in one search even if k can be larger than n.
•	Space complexity: O(1).
We only used constant auxiliary space.

             */
            public bool NaiveLinearSearch(int[] nums, int k, int t)
            {
                for (int i = 0; i < nums.Length; ++i)
                {
                    for (int j = Math.Max(i - k, 0); j < i; ++j)
                    {
                        if (Math.Abs((long)nums[i] - nums[j]) <= t) return true;
                    }
                }
                return false;
            }

            /* Approach #2 (Binary Search Tree) [Accepted]
            Complexity Analysis
            •	Time complexity: O(n log(min(n,k))).
            We iterate through the array of size n. For each iteration, it costs O(log min(k,n)) time (search, insert or delete) in the BST, since the size of the BST is upper bounded by both k and n.
            •	Space complexity: O(min(n,k)).
            Space is dominated by the size of the BST, which is upper bounded by both k and n.
•	When the array's elements and t's value are large, they can cause overflow in arithmetic operation. Consider using a larger size data type instead, such as long.
             */
            public bool UsingBinarySearchTree(int[] numbers, int k, int t)
            {
                SortedSet<int> numberSet = new SortedSet<int>();
                for (int index = 0; index < numbers.Length; ++index)
                {
                    // Find the successor of current element
                    int? successor = numberSet.GetViewBetween(numbers[index], int.MaxValue).FirstOrDefault();
                    if (successor.HasValue && (long)successor.Value <= numbers[index] + t) return true;

                    // Find the predecessor of current element
                    int? predecessor = numberSet.GetViewBetween(int.MinValue, numbers[index]).LastOrDefault();
                    if (predecessor.HasValue && numbers[index] <= (long)predecessor.Value + t) return true;

                    numberSet.Add(numbers[index]);
                    if (numberSet.Count > k)
                    {
                        numberSet.Remove(numbers[index - k]);
                    }
                }
                return false;
            }
            /* 
                        Approach #3 (Buckets) [Accepted] 
                        Complexity Analysis
            •	Time complexity: O(n).
            For each of the n elements, we do at most three searches, one insert, and one delete on the HashMap, which costs constant time on average. Thus, the entire algorithm costs O(n) time.
            •	Space complexity: O(min(n,k)).
            Space is dominated by the HashMap, which is linear to the size of its elements. The size of the HashMap is upper bounded by both n and k. Thus the space complexity is O(min(n,k)).

                        */
            public bool UsingBuckets(int[] numbers, int k, int t)
            {
                if (t < 0) return false;
                Dictionary<long, long> buckets = new Dictionary<long, long>();
                long bucketWidth = t + 1;
                for (int index = 0; index < numbers.Length; ++index)
                {
                    long bucket = GetBucketID(numbers[index], bucketWidth);
                    // check if current bucket is empty, each bucket may contain at most one element
                    if (buckets.ContainsKey(bucket)) return true;
                    // check the neighbor buckets for almost duplicate
                    if (
                        buckets.ContainsKey(bucket - 1) &&
                        Math.Abs(numbers[index] - buckets[bucket - 1]) < bucketWidth
                    ) return true;
                    if (
                        buckets.ContainsKey(bucket + 1) &&
                        Math.Abs(numbers[index] - buckets[bucket + 1]) < bucketWidth
                    ) return true;
                    // now bucket is empty and no almost duplicate in neighbor buckets
                    buckets[bucket] = numbers[index];
                    if (index >= k) buckets.Remove(GetBucketID(numbers[index - k], bucketWidth));
                }
                return false;
            }
            // Get the ID of the bucket from element value x and bucket width w
            // Java's division `/` rounds towards zero, but we need floor division for correct bucketing.
            private long GetBucketID(long elementValue, long bucketWidth)
            {
                //TODO: test below code, whether working inline to JAva's Math.floorDiv   
                return elementValue / bucketWidth; // Using integer division
            }

        }

        /* 1095. Find in Mountain Array
        https://leetcode.com/problems/find-in-mountain-array/description/
         */
        public class FindInMountainArraySol
        {
            /*             Approach 1: Binary Search 
            Complexity Analysis
            Let N be the length of the mountainArr. Moreover, let's assume that each call to mountainArr.get(k) takes O(1) time.
            •	Time complexity: O(logN)
            o	Finding the peakIndex
            There will be O(log2N) iterations in the while loop. The reason is that at each iteration, the search space is reduced to half. At each iteration, we are
            o	computing testIndex using addition and division. This takes O(1) time.
            o	calling mountainArr.get(testIndex) twice. This we assume takes O(1) time.
            o	resetting low or high. This takes O(1) time.
            Thus, the time complexity of finding the peakIndex is O(log2N).
            o	Searching in the strictly increasing part of the array
            There will be O(log2N) iterations in the while loop. The reason is that at each iteration, the search space is reduced to half. At each iteration, we are
            o	computing testIndex using addition and division. This takes O(1) time.
            o	calling mountainArr.get(testIndex) once. This we assume takes O(1) time.
            o	resetting low or high. This takes O(1) time.
            Thus, the time complexity of searching in the strictly increasing part of the array is O(log2N).
            o	Searching in the strictly decreasing part of the array
            There will be O(log2N) iterations in the while loop. The reason is that at each iteration, the search space is reduced to half. At each iteration, we are
            o	computing testIndex using addition and division. This takes O(1) time.
            o	calling mountainArr.get(testIndex) once. This we assume takes O(1) time.
            o	resetting low or high. This takes O(1) time.
            Thus, the time complexity of searching in the strictly decreasing part of the array is O(log2N).
            •	Hence, the overall time complexity of the algorithm is O(log2N).
            •	Space complexity: O(1)
            We are using only constant extra space which includes a bunch of variables. Hence, the space complexity is O(1).


            */

            public int UsingBinarySearch(int target, MountainArray mountainArr)
            {
                // Save the length of the mountain array
                int arrayLength = mountainArr.Length();

                // 1. Find the index of the peak element
                int low = 1;
                int high = arrayLength - 2;
                while (low != high)
                {
                    int testIndex = (low + high) / 2;
                    if (mountainArr.Get(testIndex) < mountainArr.Get(testIndex + 1))
                    {
                        low = testIndex + 1;
                    }
                    else
                    {
                        high = testIndex;
                    }
                }
                int peakIndex = low;

                // 2. Search in the strictly increasing part of the array
                low = 0;
                high = peakIndex;
                while (low != high)
                {
                    int testIndex = (low + high) / 2;
                    if (mountainArr.Get(testIndex) < target)
                    {
                        low = testIndex + 1;
                    }
                    else
                    {
                        high = testIndex;
                    }
                }
                // Check if the target is present in the strictly increasing part
                if (mountainArr.Get(low) == target)
                {
                    return low;
                }

                // 3. Otherwise, search in the strictly decreasing part
                low = peakIndex + 1;
                high = arrayLength - 1;
                while (low != high)
                {
                    int testIndex = (low + high) / 2;
                    if (mountainArr.Get(testIndex) > target)
                    {
                        low = testIndex + 1;
                    }
                    else
                    {
                        high = testIndex;
                    }
                }
                // Check if the target is present in the strictly decreasing part
                if (mountainArr.Get(low) == target)
                {
                    return low;
                }

                // Target is not present in the mountain array
                return -1;
            }
            public class MountainArray
            {
                internal int Get(int v)
                {
                    throw new NotImplementedException();
                }

                internal int Length()
                {
                    throw new NotImplementedException();
                }
            }
            /*             Approach 2: Minimizing get Calls with Early Stopping and Caching

Complexity Analysis
Let N be the length of the mountainArr. Moreover, let's assume that each call to mountainArr.get(k) takes O(1) time.
•	Time complexity: O(logN)
o	Finding the peakIndex
There will be O(log2N) iterations in the while loop. The reason is that at each iteration, the search space is reduced to half. At each iteration, we are
o	computing testIndex using addition and bit shift. This takes O(1) time.
o	Getting the value of mountainArr.get(testIndex) from the cache or from the mountainArr. Caching if not present in the cache. This takes O(1) time.
o	Getting the value of mountainArr.get(testIndex + 1) from the cache or from the mountainArr. Caching if not present in the cache. This takes O(1) time.
o	Returning or resetting low or high. This takes O(1) time.
Thus, the time complexity of finding the peakIndex is O(log2N).
o	Searching in the strictly increasing part of the array
There will be O(log2N) iterations in the while loop. The reason is that at each iteration, the search space is reduced to half. At each iteration, we are
o	computing testIndex using addition and bit shift. This takes O(1) time.
o	Getting the value of mountainArr.get(testIndex) from the cache or from the mountainArr. This takes O(1) time.
o	Returning or resetting low or high. This takes O(1) time.
Thus, the time complexity of searching in the strictly increasing part of the array is O(log2N).
o	Searching in the strictly decreasing part of the array
There will be O(log2N) iterations in the while loop. The reason is that at each iteration, the search space is reduced to half. At each iteration, we are
o	computing testIndex using addition and bit shift. This takes O(1) time.
o	Getting the value of mountainArr.get(testIndex) from the cache or from the mountainArr. This takes O(1) time.
o	Returning or resetting low or high. This takes O(1) time.
Thus, the time complexity of searching in the strictly decreasing part of the array is O(log2N).
•	Hence, the overall time complexity is O(log2N).
•	Space complexity: O(logN)
The cache will contain O(logN) elements because we are caching only the elements for which we are calling mountainArr.get(k).
Hence, the space complexity is O(logN).

             */
            public int FindInMountainArray(int target, MountainArray mountainArr)
            {
                // Save the length of the mountain array
                int length = mountainArr.Length();

                // Initialize the cache
                Dictionary<int, int> cache = new Dictionary<int, int>();

                // 1. Find the index of the peak element
                int low = 1;
                int high = length - 2;
                while (low != high)
                {
                    int testIndex = (low + high) >> 1;

                    int curr;
                    if (cache.ContainsKey(testIndex))
                    {
                        curr = cache[testIndex];
                    }
                    else
                    {
                        curr = mountainArr.Get(testIndex);
                        cache[testIndex] = curr;
                    }

                    int next;
                    if (cache.ContainsKey(testIndex + 1))
                    {
                        next = cache[testIndex + 1];
                    }
                    else
                    {
                        next = mountainArr.Get(testIndex + 1);
                        cache[testIndex + 1] = next;
                    }

                    if (curr < next)
                    {
                        if (curr == target)
                        {
                            return testIndex;
                        }
                        if (next == target)
                        {
                            return testIndex + 1;
                        }
                        low = testIndex + 1;
                    }
                    else
                    {
                        high = testIndex;
                    }
                }

                int peakIndex = low;

                // 2. Search in the strictly increasing part of the array
                // If found, will be returned in the loop itself.
                low = 0;
                high = peakIndex;
                while (low <= high)
                {
                    int testIndex = (low + high) >> 1;

                    int curr;
                    if (cache.ContainsKey(testIndex))
                    {
                        curr = cache[testIndex];
                    }
                    else
                    {
                        curr = mountainArr.Get(testIndex);
                    }

                    if (curr == target)
                    {
                        return testIndex;
                    }
                    else if (curr < target)
                    {
                        low = testIndex + 1;
                    }
                    else
                    {
                        high = testIndex - 1;
                    }
                }

                // 3. Search in the strictly decreasing part of the array
                // If found, will be returned in the loop itself.
                low = peakIndex + 1;
                high = length - 1;
                while (low <= high)
                {
                    int testIndex = (low + high) >> 1;

                    int curr;
                    if (cache.ContainsKey(testIndex))
                    {
                        curr = cache[testIndex];
                    }
                    else
                    {
                        curr = mountainArr.Get(testIndex);
                    }

                    if (curr == target)
                    {
                        return testIndex;
                    }
                    else if (curr > target)
                    {
                        high = testIndex - 1;
                    }
                    else
                    {
                        low = testIndex + 1;
                    }
                }

                // Target is not present in the mountain array
                return -1;
            }


        }


        /* 952. Largest Component Size by Common Factor
        https://leetcode.com/problems/largest-component-size-by-common-factor/description/
         */
        public class LargestComponentSizeSol
        {

            /* 
            Approach 1: Union-Find via Factors
            Complexity Analysis
            Since we applied the Union-Find data structure in our algorithm, we would like to start with a statement on the time complexity of the data structure, as follows:
            Statement: If M operations, either Union or Find, are applied to N elements, the total run time is O(M⋅log∗N), where log∗ is the iterated logarithm.
            One can refer to the proof of Union-Find complexity for more details.
            In our case, the number of elements in the Union-Find data structure is equal to the maximum number of the input list, i.e. max(A).
            Let N be the number of elements in the input list, and M be the maximum value of the input list.
            •	Time Complexity: O(N⋅sqrt of(M)⋅(log^∗) . M)
            o	The number of factors for a given number is bounded by O(sqrt of(M)). Assuming that any number that is less than M can be divided by M, we would then have 2⋅ sqrt of(M) pairs of factors.
            o	In the first step, we iterate through each number (i.e. N iterations), and for each number, we iterate through all its factors (i.e. up to 2⋅ sqrt of(M) iterations). As a result, the time complexity of this step would be O(N⋅ sqrt of(M)⋅(log^∗)M).
            o	In the second step, we iterate through each number again.
            But this time, for each iteration we perform only once the Union-Find operation.
            Hence, the time complexity for this step would be O(N⋅log^∗M).
            o	To sum up, the overall complexity of the algorithm would be O(N⋅ sqrt of(M)⋅(log^∗)M)+O(N⋅(log^∗)M)=O(N⋅ sqrt of(M)⋅(log^∗)M).
            •	Space Complexity: O(M+N)
            o	The space complexity of the Union-Find data structure is O(M).
            o	In the main algorithm, we use a hash table to keep track of the account for each group. In the worst case, each number forms an individual group. Therefore, the space complexity of this hash table is O(N).
            o	To sum up, the overall space complexity of the algorithm is O(M)+O(N)=O(M+N).


             */
            public int UnionFindViaFactors(int[] A)
            {
                int maxValue = 0;
                foreach (var num in A)
                {
                    maxValue = Math.Max(maxValue, num);
                }

                DisjointSetUnion dsu = new DisjointSetUnion(maxValue);

                // attribute each element to all the groups that lead by its factors.
                foreach (var num in A)
                {
                    for (int factor = 2; factor <= Math.Sqrt(num); ++factor)
                    {
                        if (num % factor == 0)
                        {
                            dsu.Union(num, factor);
                            dsu.Union(num, num / factor);
                        }
                    }
                }

                // count the size of group one by one
                int maxGroupSize = 0;
                Dictionary<int, int> groupCount = new Dictionary<int, int>();
                foreach (var num in A)
                {
                    int groupId = dsu.Find(num);
                    if (!groupCount.ContainsKey(groupId))
                    {
                        groupCount[groupId] = 0;
                    }
                    groupCount[groupId]++;
                    maxGroupSize = Math.Max(maxGroupSize, groupCount[groupId]);
                }

                return maxGroupSize;
            }
            public class DisjointSetUnion
            {
                private int[] parent;
                private int[] rank;

                public DisjointSetUnion(int size)
                {
                    parent = new int[size + 1];
                    rank = new int[size + 1];
                    for (int i = 0; i <= size; i++)
                    {
                        parent[i] = i;
                        rank[i] = 1;
                    }
                }

                /** return the component id that the element x belongs to. */
                public int Find(int x)
                {
                    if (parent[x] != x)
                    {
                        parent[x] = Find(parent[x]);
                    }
                    return parent[x];
                }
                /**
                 * merge the two components that x, y belongs to respectively,
                 * and return the merged component id as the result.
                 */
                public int Union(int x, int y)
                {
                    int px = Find(x);
                    int py = Find(y);
                    // the two nodes share the same group
                    if (px == py)
                        return px;

                    // otherwise, connect the two sets (components)
                    if (this.rank[px] > this.rank[py])
                    {
                        // add the node to the union with less members.
                        // keeping px as the index of the smaller component
                        int temp = px;
                        px = py;
                        py = temp;
                    }

                    // add the smaller component to the larger one
                    this.parent[px] = py;
                    this.rank[py] += this.rank[px];
                    return py;
                }
            }
            /* 
            Approach 2: Union-Find on Prime Factors
Complexity Analysis
Let N be the number of elements in the input list, and M be the maximum value of the input list.
•	Time Complexity: O(N⋅(log2M⋅(log^∗)M+SqrtOf(M)))
o	First of all, the time complexity of the sieve method to calculate the prime factors of is O(SqrtOf(M)).
o	It is hard to estimate the number of prime factors for a given number. Since the smallest prime number is 2, a coarse upper bound for the number of the prime factors is log2M, e.g. 8=2∗2∗2.
o	In the first step, we iterate through each number (i.e. N iterations), and for each number, we iterate through all its factors (i.e. up to log2M iterations). As a result, together with the calculation of prime factors, the time complexity of this step would be O(N⋅log2M⋅log^∗ SqrtOf(M))+O(N⋅ SqrtOf(M))=O(N⋅(log2M⋅log∗M+ SqrtOf(M))).
o	In the second step, we iterate through each number again.
But this time, for each iteration we perform only once the Union-Find operation, i.e. O(N⋅log∗ SqrtOf(M)).
o	To sum up, the overall complexity of the algorithm would be O(N⋅(log2M⋅log∗M+ SqrtOf(M))).
o	As one might notice that, the asymptotic complexity of this approach seems to be inferior than the previous approach, due to the calculation of prime factors. However, in reality, the saving we gain on the Union-Find operations could outweigh the cost of prime factor calculation.
•	Space Complexity: O(M+N)
o	The space complexity of the Union-Find data structure is O(M).
o	In the main algorithm, we use a hash table to keep track of the count for each group. In the worst case, each number forms an individual group. Therefore, the space complexity of this hash table is O(N).
o	In addition, we keep a map between each number and one of its prime factors. Hence the space complexity of this map is O(N).
o	To sum up, the overall space complexity of the algorithm is O(M)+O(N)+O(N)=O(M+N).

             */
            public int UnionFindOnPrimeFactors(int[] A)
            {
                int maxValue = A.Max();
                DisjointSetUnion dsu = new DisjointSetUnion(maxValue);

                Dictionary<int, int> numFactorMap = new Dictionary<int, int>();

                // Union-Find on the prime factors.
                foreach (int num in A)
                {
                    // find all the unique prime factors.
                    List<int> primeFactors = new HashSet<int>(PrimeDecompose(num)).ToList();

                    // map a number to its first prime factor
                    numFactorMap[num] = primeFactors[0];
                    // Merge all the groups that contain the prime factors.
                    for (int i = 0; i < primeFactors.Count - 1; ++i)
                        dsu.Union(primeFactors[i], primeFactors[i + 1]);
                }

                // count the size of group one by one
                int maxGroupSize = 0;
                Dictionary<int, int> groupCount = new Dictionary<int, int>();
                foreach (int num in A)
                {
                    int groupId = dsu.Find(numFactorMap[num]);
                    int count = groupCount.GetValueOrDefault(groupId, 0);
                    groupCount[groupId] = count + 1;
                    maxGroupSize = Math.Max(maxGroupSize, count + 1);
                }

                return maxGroupSize;
            }

            protected List<int> PrimeDecompose(int num)
            {
                List<int> primeFactors = new List<int>();
                int factor = 2;
                while (num >= factor * factor)
                {
                    if (num % factor == 0)
                    {
                        primeFactors.Add(factor);
                        num /= factor;
                    }
                    else
                    {
                        factor += 1;
                    }
                }
                primeFactors.Add(num);
                return primeFactors;
            }

        }


        /* 2009. Minimum Number of Operations to Make Array Continuous
        https://leetcode.com/problems/minimum-number-of-operations-to-make-array-continuous/description/
         */

        public class MinOperationsToMakeArrayContinuousSol
        {

            /* Approach 1: Binary Search
            Complexity Analysis
            Given n as the length of nums,
            •	Time complexity: O(n⋅logn)
            To remove duplicates and sort nums, we require O(n⋅logn) time.
            Then, we iterate over n indices and perform a O(logn) binary search at each index.
            •	Space complexity: O(n)
            We create a new array newNums of size O(n). Note that even if you were to modify the input directly, we still use O(n) space creating a hash set to remove duplicates. Also, it is considered a bad practice to modify the input, and many people will argue that modifying the input makes it part of the space complexity anyway.

             */
            public int UsingBinarySearch(int[] nums)
            {
                int arrayLength = nums.Length;
                int minimumOperations = arrayLength;

                HashSet<int> uniqueNumbers = new HashSet<int>();
                foreach (int number in nums)
                {
                    uniqueNumbers.Add(number);
                }

                int[] newNumbers = new int[uniqueNumbers.Count];
                int index = 0;

                foreach (int number in uniqueNumbers)
                {
                    newNumbers[index++] = number;
                }

                Array.Sort(newNumbers);

                for (int i = 0; i < newNumbers.Length; i++)
                {
                    int leftBoundary = newNumbers[i];
                    int rightBoundary = leftBoundary + arrayLength - 1;
                    int position = BinarySearch(newNumbers, rightBoundary);
                    int count = position - i;
                    minimumOperations = Math.Min(minimumOperations, arrayLength - count);
                }

                return minimumOperations;
            }

            private int BinarySearch(int[] newNumbers, int target)
            {
                int leftIndex = 0;
                int rightIndex = newNumbers.Length;

                while (leftIndex < rightIndex)
                {
                    int middleIndex = (leftIndex + rightIndex) / 2;
                    if (target < newNumbers[middleIndex])
                    {
                        rightIndex = middleIndex;
                    }
                    else
                    {
                        leftIndex = middleIndex + 1;
                    }
                }

                return leftIndex;
            }

            /* Approach 2: Sliding Window
            Complexity Analysis
            Given n as the length of nums,
            •	Time complexity: O(n⋅logn)
            To remove duplicates and sort nums, we require O(n⋅logn) time.
            Then, we iterate over n indices and perform O(1) amortized work at each iteration. The while loop inside the for loop can only iterate at most n times total across all iterations of the for loop. Each element in newNums can only be iterated over once by this while loop.
            Despite this approach having the same time complexity as the previous approach (due to the sort), it is a slight practical improvement as the sliding window portion is O(n).
            •	Space complexity: O(n)
            We create a new array newNums of size O(n). Note that even if you were to modify the input directly, we still use O(n) space creating a hash set to remove duplicates. Also, it is considered a bad practice to modify the input, and many people will argue that modifying the input makes it part of the space complexity anyway.

             */
            public int UsingSlidingWindow(int[] nums)
            {
                int n = nums.Length;
                int ans = n;

                HashSet<int> unique = new HashSet<int>();
                foreach (int num in nums)
                {
                    unique.Add(num);
                }

                int[] newNums = new int[unique.Count];
                int index = 0;

                foreach (int num in unique)
                {
                    newNums[index++] = num;
                }

                Array.Sort(newNums);

                int j = 0;
                for (int i = 0; i < newNums.Length; i++)
                {
                    while (j < newNums.Length && newNums[j] < newNums[i] + n)
                    {
                        j++;
                    }

                    int count = j - i;
                    ans = Math.Min(ans, n - count);
                }

                return ans;
            }

        }


        /* 1187. Make Array Strictly Increasing
        https://leetcode.com/problems/make-array-strictly-increasing/description/
         */
        class MakeArrayIncreasingSol
        {
            /* Approach 1: Top-down Dynamic Programming
            Complexity Analysis
            Let m,n be the length of arr1 and arr2.
            •	Time complexity: O(m⋅n⋅logn)
            o	Sorting arr2 takes O(nlogn) time.
            o	To improve the efficiency of the algorithm, we use memoization and store the minimum number of operations to reach each state (i, prev) in a hash map dp. There are m indices and at most n+1 possible prev as we might replace arr[i] with any value in arr2. Each state is computed with a binary search over arr2, which takes O(logn).
            •	Space complexity: O(m⋅n)
            o	The maximum number of distinct states in dp is m⋅n.

             */
            public int TopDownDP(int[] arr1, int[] arr2)
            {
                Array.Sort(arr2);

                int answer = Dfs(0, -1, arr1, arr2);

                return answer < 2001 ? answer : -1;
            }

            private Dictionary<(int, int), int> dp = new Dictionary<(int, int), int>();

            private int Dfs(int index, int previous, int[] arr1, int[] arr2)
            {
                if (index == arr1.Length)
                {
                    return 0;
                }
                if (dp.TryGetValue((index, previous), out int cachedValue))
                {
                    return cachedValue;
                }

                int cost = 2001;

                // If arr1[index] is already greater than previous, we can leave it be.
                if (arr1[index] > previous)
                {
                    cost = Dfs(index + 1, arr1[index], arr1, arr2);
                }

                // Find a replacement with the smallest value in arr2.
                int idx = BisectRight(arr2, previous);

                // Replace arr1[index], with a cost of 1 operation.
                if (idx < arr2.Length)
                {
                    cost = Math.Min(cost, 1 + Dfs(index + 1, arr2[idx], arr1, arr2));
                }

                dp[(index, previous)] = cost;
                return cost;
            }

            private static int BisectRight(int[] arr, int value)
            {
                int left = 0, right = arr.Length;
                while (left < right)
                {
                    int mid = (left + right) / 2;
                    if (arr[mid] <= value)
                    {
                        left = mid + 1;
                    }
                    else
                    {
                        right = mid;
                    }
                }
                return left;
            }
            /* 
            Approach 2: Bottom-up Dynamic Programming
            Complexity Analysis
Let m,n be the length of arr1 and arr2.
•	Time complexity: O(m⋅n⋅logn)
o	Sorting arr2 takes O(nlogn) time.
o	We update dp by m rounds. In each round at index i, there are at most n+1 possible prev as we might replace arr[i] with any of the n values in arr2 or leave it unchanged. Each state is computed with a binary search over all start times, which takes O(logn).
•	Space complexity: O(n)
o	We keep track of all states (i, prev) of two latest indices in dp and new_dp, respectively. At each index, the number of possible distinct states is at most n+1

             */
            public int BottomUpDP(int[] arr1, int[] arr2)
            {
                Dictionary<int, int> dynamicProgrammingMap = new Dictionary<int, int>();
                Array.Sort(arr2);
                int array2Length = arr2.Length;
                dynamicProgrammingMap[-1] = 0;

                for (int i = 0; i < arr1.Length; i++)
                {
                    Dictionary<int, int> newDynamicProgrammingMap = new Dictionary<int, int>();
                    foreach (int previousValue in dynamicProgrammingMap.Keys)
                    {
                        if (arr1[i] > previousValue)
                        {
                            newDynamicProgrammingMap[arr1[i]] = Math.Min(newDynamicProgrammingMap.GetValueOrDefault(arr1[i], int.MaxValue), dynamicProgrammingMap[previousValue]);
                        }
                        int index = BisectRight(arr2, previousValue);
                        if (index < array2Length)
                        {
                            newDynamicProgrammingMap[arr2[index]] = Math.Min(newDynamicProgrammingMap.GetValueOrDefault(arr2[index], int.MaxValue), 1 + dynamicProgrammingMap[previousValue]);
                        }
                    }
                    dynamicProgrammingMap = newDynamicProgrammingMap;
                }

                int minimumTransformations = int.MaxValue;
                foreach (int value in dynamicProgrammingMap.Values)
                {
                    minimumTransformations = Math.Min(minimumTransformations, value);
                }

                return minimumTransformations == int.MaxValue ? -1 : minimumTransformations;
            }
        }


        /* 1269. Number of Ways to Stay in the Same Place After Some Steps
        https://leetcode.com/problems/number-of-ways-to-stay-in-the-same-place-after-some-steps/description/
         */

        class NumWaysToStayInSamePlaceAfterSomeStepsSol
        {
            int[][] memo;
            int MOD = (int)1e9 + 7;
            int arrLen;

            /* Approach 1: Top-Down Dynamic Programming
            Complexity Analysis
            Given n as steps and m as arrLen,
            •	Time complexity: O(n⋅min(n,m))
            There can be steps values of remain and min(steps, arrLen) values of curr. The reason curr is limited by steps is because if we were to only move right, we would eventually run out of moves. Thus, there are O(n⋅min(n,m)) states of curr, remain. Due to memoization, we never calculate a state more than once. To calculate a given state costs O(1) as we are simply adding up three options.
            •	Space complexity: O(n⋅min(n,m))
            The recursion call stack uses up to O(n) space, but this is dominated by memo which has a size of O(n⋅min(n,m)).

             */
            public int TopDownDP(int steps, int arrLen)
            {
                arrLen = Math.Min(arrLen, steps);
                this.arrLen = arrLen;
                memo = new int[arrLen][];
                foreach (int[] row in memo)
                {
                    Array.Fill(row, -1);
                }

                return Dp(0, steps);
            }
            public int Dp(int curr, int remain)
            {
                if (remain == 0)
                {
                    if (curr == 0)
                    {
                        return 1;
                    }

                    return 0;
                }

                if (memo[curr][remain] != -1)
                {
                    return memo[curr][remain];
                }

                int ans = Dp(curr, remain - 1);
                if (curr > 0)
                {
                    ans = (ans + Dp(curr - 1, remain - 1)) % MOD;
                }

                if (curr < arrLen - 1)
                {
                    ans = (ans + Dp(curr + 1, remain - 1)) % MOD;
                }

                memo[curr][remain] = ans;
                return ans;
            }

            /* Approach 2: Bottom-Up Dynamic Programming
            Complexity Analysis
            Given n as steps and m as arrLen,
            •	Time complexity: O(n⋅min(n,m))
            Our nested for-loops iterate over O(n⋅min(n,m)) states of curr, remain. Calculating each state is done in O(1).
            •	Space complexity: O(n⋅min(n,m))
            dp has a size of O(n⋅min(n,m)).

             */
            public int BottomUpDP(int steps, int arrLen)
            {
                int MOD = (int)1e9 + 7;
                arrLen = Math.Min(arrLen, steps);
                int[][] dp = new int[arrLen][];
                dp[0][0] = 1;

                for (int remain = 1; remain <= steps; remain++)
                {
                    for (int curr = arrLen - 1; curr >= 0; curr--)
                    {
                        int ans = dp[curr][remain - 1];

                        if (curr > 0)
                        {
                            ans = (ans + dp[curr - 1][remain - 1]) % MOD;
                        }

                        if (curr < arrLen - 1)
                        {
                            ans = (ans + dp[curr + 1][remain - 1]) % MOD;
                        }

                        dp[curr][remain] = ans;
                    }
                }

                return dp[0][steps];
            }
            /*             Approach 3: Space-Optimized Dynamic Programming
            Complexity Analysis
            Given n as steps and m as arrLen,
            •	Time complexity: O(n⋅min(n,m))
            Our nested for-loops iterate over O(n⋅min(n,m)) states of curr, remain. Calculating each state is done in O(1).
            •	Space complexity: O(min(n,m))
            dp and prevDp have a size of O(min(n,m)).	

             */
            public int BottomUpDPSpaceOptimal(int steps, int arrLen)
            {
                int MOD = (int)1e9 + 7;
                arrLen = Math.Min(arrLen, steps);
                int[] dp = new int[arrLen];
                int[] prevDp = new int[arrLen];
                prevDp[0] = 1;

                for (int remain = 1; remain <= steps; remain++)
                {
                    dp = new int[arrLen];

                    for (int curr = arrLen - 1; curr >= 0; curr--)
                    {
                        int ans = prevDp[curr];
                        if (curr > 0)
                        {
                            ans = (ans + prevDp[curr - 1]) % MOD;
                        }

                        if (curr < arrLen - 1)
                        {
                            ans = (ans + prevDp[curr + 1]) % MOD;
                        }

                        dp[curr] = ans;
                    }

                    prevDp = dp;
                }

                return dp[0];
            }
        }

        /* 1675. Minimize Deviation in Array
        https://leetcode.com/problems/minimize-deviation-in-array/description/
         */

        public class MinmizeDeviationInArraySol
        {
            /* Approach 1: Simulation + Heap
            Complexity Analysis
Let N be the length of nums, and M be the largest number in nums. In the worst case when M is the power of 2, there are log(M) possible values for M. Therefore, in the worst case, the total possible candidate number (denoted by K) is K=N⋅log(M)=Nlog(M).
•	Time Complexity: O(Klog(N))=O(Nlog(M)log(N)). In worst case, we need to push every candidate number into evens, and the size of evens is O(N). Hence, the total time complexity is O(K⋅log(N))=O(Nlog(M)log(N)).
•	Space Complexity: O(N), since there are at most N elements in evens.

             */
            public int SimulationAndMaxHeap(int[] nums)
            {
                //MaxHeap
                var evens = new PriorityQueue<int, int>(Comparer<int>.Create((x, y) => y.CompareTo(x)));
                int minimum = int.MaxValue;
                foreach (int num in nums)
                {
                    if (num % 2 == 0)
                    {
                        evens.Enqueue(num, num);
                        minimum = Math.Min(minimum, num);
                    }
                    else
                    {
                        evens.Enqueue(num * 2, num * 2);
                        minimum = Math.Min(minimum, num * 2);
                    }
                }
                int minDeviation = int.MaxValue;

                while (evens.Count > 0)
                {
                    int currentValue = evens.Dequeue();
                    minDeviation = Math.Min(minDeviation, currentValue - minimum);
                    if (currentValue % 2 == 0)
                    {
                        evens.Enqueue(currentValue / 2, currentValue / 2);
                        minimum = Math.Min(minimum, currentValue / 2);
                    }
                    else
                    {
                        // if the maximum is odd, break and return
                        break;
                    }
                }
                return minDeviation;
            }
            /* Approach 2: Pretreatment + Sorting + Sliding Window
Complexity Analysis
Let N be the length of nums, and M be the largest number in nums. In the worst case when M is the power of 2, there are log(M) candidates for M. Therefore, in the worst case, the total candidate number (denoted by K) is K=N⋅log(M)=Nlog(M).
•	Time Complexity: O(Klog(K))=O(Nlog(M)log(Nlog(M))). In the worst case, possible has K elements, and we need to sort it, which costs O(Klog(K)). For the sliding window, we need O(K) time, since the left pointer and the right pointer visit every element in possible once.
•	Space Complexity: O(K)=O(Nlog(M)), since in the worst case, possible has K elements.

             */
            public int PretreatmentWithSortingAndSlidingWindow(int[] nums)
            {
                int n = nums.Length;
                List<int[]> possible = new List<int[]>();
                // pretreatment
                for (int i = 0; i < n; i++)
                {
                    if (nums[i] % 2 == 0)
                    {
                        int temp = nums[i];
                        possible.Add(new int[] { temp, i });
                        while (temp % 2 == 0)
                        {
                            temp /= 2;
                            possible.Add(new int[] { temp, i });
                        }
                    }
                    else
                    {
                        possible.Add(new int[] { nums[i], i });
                        possible.Add(new int[] { nums[i] * 2, i });
                    }
                }
                possible.Sort((p1, p2) => p1[0].CompareTo(p2[0]));
                // start sliding window
                int minDeviation = int.MaxValue;
                int notIncluded = n;
                int currentStart = 0;
                int[] needInclude = new int[n];
                for (int i = 0; i < n; i++)
                {
                    needInclude[i] = 1;
                }
                foreach (int[] current in possible)
                {
                    int currentValue = current[0];
                    int currentItem = current[1];
                    needInclude[currentItem] -= 1;
                    if (needInclude[currentItem] == 0)
                    {
                        notIncluded -= 1;
                    }
                    if (notIncluded == 0)
                    {
                        while (needInclude[possible[currentStart][1]] < 0)
                        {
                            needInclude[possible[currentStart][1]] += 1;
                            currentStart += 1;
                        }
                        if (minDeviation > currentValue - possible[currentStart][0])
                        {
                            minDeviation = currentValue - possible[currentStart][0];
                        }
                        needInclude[possible[currentStart][1]] += 1;
                        currentStart += 1;
                        notIncluded += 1;
                    }
                }
                return minDeviation;
            }
            /* 
                        Approach 3: Pretreatment + Heap + Sliding Window
            Complexity Analysis
            Let N be the length of nums, and M be the largest number in nums. In the worst case when M is the power of 2, there are log(M) candidates for M. Therefore, in the worst case, the total candidate number (denoted by K) is K=N⋅log(M)=Nlog(M).
            •	Time Complexity: O(Klog(K))=O(Nlog(M)log(Nlog(M))). In the worst case, possible has K elements, and we need to pop every elements from it, which costs O(Klog(K)). For the sliding window, we need O(K) time, since the left pointer and the right pointer visit every element in possible once.
            •	Space Complexity: O(K)=O(Nlog(M)), since in the worst case, possible has K elements.

                         */
            public int PretreatmentWithMinHeapAndSlidingWindow(int[] nums)
            {
                int n = nums.Length;
                // pretreatment
                List<int[]> possible = new List<int[]>();

                for (int i = 0; i < n; i++)
                {
                    if (nums[i] % 2 == 0)
                    {
                        int temp = nums[i];
                        possible.Add(new int[] { temp, i });
                        while (temp % 2 == 0)
                        {
                            temp /= 2;
                            possible.Add(new int[] { temp, i });
                        }
                    }
                    else
                    {
                        possible.Add(new int[] { nums[i], i });
                        possible.Add(new int[] { nums[i] * 2, i });
                    }
                }
                PriorityQueue<int[], int[]> minHeap = new PriorityQueue<int[], int[]>(Comparer<int[]>.Create((a, b) => a[0].CompareTo(b[0])));
                foreach (var item in possible)
                {
                    minHeap.Enqueue(new int[] { item[0], item[1] }, new int[] { item[0], item[1] });
                }
                int minDeviation = int.MaxValue;
                int notIncluded = n;
                int currentStart = 0;
                int[] needInclude = new int[n];
                for (int i = 0; i < n; i++)
                {
                    needInclude[i] = 1;
                }
                List<int[]> seen = new List<int[]>();
                // get minimum from heap
                while (minHeap.Count > 0)
                {
                    int[] current = minHeap.Dequeue();
                    seen.Add(current);
                    int currentValue = current[0];
                    int currentItem = current[1];
                    needInclude[currentItem] -= 1;
                    if (needInclude[currentItem] == 0)
                    {
                        notIncluded -= 1;
                    }
                    if (notIncluded == 0)
                    {
                        while (needInclude[seen[currentStart][1]] < 0)
                        {
                            needInclude[seen[currentStart][1]] += 1;
                            currentStart += 1;
                        }
                        if (minDeviation > currentValue - seen[currentStart][0])
                        {
                            minDeviation = currentValue - seen[currentStart][0];
                        }
                        needInclude[seen[currentStart][1]] += 1;
                        currentStart += 1;
                        notIncluded += 1;
                    }
                }
                return minDeviation;
            }
            /* Approach 4: Pretreatment + Heap + Pointers
            Complexity Analysis
Let N be the length of nums, and M be the largest number in nums. In the worst case when M is the power of 2, there are log(M) candidates for M. Therefore, in the worst case, the total candidate number (denoted by K) is K=N⋅log(M)=Nlog(M).
•	Time Complexity: O(Klog(N))=O(Nlog(M)log(N)). In the worst case, possibleList has K candidates, and we need to push every candidates into pointers, which cost O(Klog(N)).
•	Space Complexity: O(N), since pointers always has N elements.

             */
            public int PretreatmentWithMinHeapAndPointers(int[] nums)
            {
                // pretreatment
                int n = nums.Length;
                List<List<int>> possibleList = new List<List<int>>();
                for (int i = 0; i < n; i++)
                {
                    List<int> candidates = new List<int>();
                    if (nums[i] % 2 == 0)
                    {
                        int temp = nums[i];
                        candidates.Add(temp);
                        while (temp % 2 == 0)
                        {
                            temp /= 2;
                            candidates.Add(temp);
                        }
                    }
                    else
                    {
                        candidates.Add(nums[i] * 2);
                        candidates.Add(nums[i]);
                    }
                    // reverse candidates to sort from small to large
                    candidates.Reverse();
                    possibleList.Add(candidates);
                }
                List<int[]> pointers = new List<int[]>();
                for (int i = 0; i < n; i++)
                {
                    pointers.Add(new int[] { possibleList[i][0], i, 0 });
                }
                PriorityQueue<int[], int[]> minHeap = new PriorityQueue<int[], int[]>(Comparer<int[]>.Create((a, b) => a[0].CompareTo(b[0])));
                foreach (var pointer in pointers)
                {
                    minHeap.Enqueue(pointer, pointer);
                }

                int minDeviation = int.MaxValue;
                int currentEnd = 0;
                for (int i = 0; i < n; i++)
                {
                    currentEnd = Math.Max(currentEnd, possibleList[i][0]);
                }
                // get minimum from heap
                while (minHeap.Count > 0)
                {
                    int[] current = minHeap.Dequeue();
                    int currentStart = current[0];
                    int index = current[1];
                    int indexInCandidates = current[2];
                    if (minDeviation > currentEnd - currentStart)
                    {
                        minDeviation = currentEnd - currentStart;
                    }
                    // if the pointer reach last
                    if (indexInCandidates + 1 == possibleList[index].Count)
                    {
                        return minDeviation;
                    }
                    int nextValue = possibleList[index][indexInCandidates + 1];
                    currentEnd = Math.Max(currentEnd, nextValue);
                    minHeap.Enqueue(new int[] { nextValue, index, indexInCandidates + 1 }, new int[] { nextValue, index, indexInCandidates + 1 });
                }
                return minDeviation;
            }

            /* Approach 5: Pretreatment + Sorting + Pointers
            Complexity Analysis
Let N be the length of nums, and M be the largest number in nums. In the worst case when M is the power of 2, there are log(M) candidates for M. Therefore, in the worst case, the total candidate number (denoted by K) is K=N⋅log(M)=Nlog(M).
•	Time Complexity: O(Klog(K))=O(Nlog(M)log(Nlog(M))). In the worst case, possibleList has K candidates, and we need to sort K pointers, which cost O(Klog(K)).
•	Space Complexity: O(K), since pointers has K elements in the worst case.

             */
            public int PretreatmentWithSortingAndPointers(int[] nums)
            {
                int n = nums.Length;
                // pretreatment
                List<List<int>> possibleList = new List<List<int>>();
                for (int i = 0; i < n; i++)
                {
                    List<int> candidates = new List<int>();
                    if (nums[i] % 2 == 0)
                    {
                        int temp = nums[i];
                        candidates.Add(temp);
                        while (temp % 2 == 0)
                        {
                            temp /= 2;
                            candidates.Add(temp);
                        }
                    }
                    else
                    {
                        candidates.Add(nums[i] * 2);
                        candidates.Add(nums[i]);
                    }
                    // reverse candidates to sort from small to large
                    candidates.Reverse();
                    possibleList.Add(candidates);
                }
                List<int[]> pointers = new List<int[]>();
                for (int i = 0; i < n; i++)
                {
                    int size = possibleList[i].Count;
                    for (int j = 0; j < size; j++)
                    {
                        pointers.Add(new int[] { possibleList[i][j], i, j });
                    }
                }
                pointers.Sort((p1, p2) => p1[0].CompareTo(p2[0]));
                int minDeviation = int.MaxValue;
                int currentEnd = 0;
                for (int i = 0; i < n; i++)
                {
                    currentEnd = Math.Max(currentEnd, possibleList[i][0]);
                }
                foreach (int[] current in pointers)
                {
                    int currentStart = current[0];
                    int index = current[1];
                    int indexInCandidates = current[2];
                    if (minDeviation > currentEnd - currentStart)
                    {
                        minDeviation = currentEnd - currentStart;
                    }
                    // if the pointer reach last
                    if (indexInCandidates + 1 == possibleList[index].Count)
                    {
                        return minDeviation;
                    }
                    int nextValue = possibleList[index][indexInCandidates + 1];
                    currentEnd = Math.Max(currentEnd, nextValue);
                }
                return minDeviation;
            }

        }

        /* 1420. Build Array Where You Can Find The Maximum Exactly K Comparisons
        https://leetcode.com/problems/build-array-where-you-can-find-the-maximum-exactly-k-comparisons/description/
         */

        class NumOfArraysSol
        {
            int[][][] memo;
            int MOD = (int)1e9 + 7;
            int n;
            int m;
            /* Approach 1: Top-Down Dynamic Programming
            Complexity Analysis
            •	Time complexity: O(n⋅m^2⋅k)
            There are n⋅m⋅k possible states of dp. Because of memoization, we never calculate a state more than once. To calculate a given state, we have for loops that iterate O(m) times. Thus, to calculate O(n⋅m⋅k) states costs O(n⋅m^2⋅k).
            •	Space complexity: O(n⋅m⋅k)
            The recursion call stack uses some space, but it will be dominated by the memoization of dp. We are storing the results of O(n⋅m⋅k) states.

             */
            public int TopDownDP(int n, int m, int k)
            {
                memo = new int[n][][];
                for (int i = 0; i < n; i++)
                {
                    for (int j = 0; j <= m; j++)
                    {
                        Array.Fill(memo[i][j], -1);
                    }
                }

                this.n = n;
                this.m = m;
                return dp(0, 0, k);
            }

            public int dp(int i, int maxSoFar, int remain)
            {
                if (i == n)
                {
                    if (remain == 0)
                    {
                        return 1;
                    }

                    return 0;
                }

                if (remain < 0)
                {
                    return 0;
                }

                if (memo[i][maxSoFar][remain] != -1)
                {
                    return memo[i][maxSoFar][remain];
                }

                int ans = 0;
                for (int num = 1; num <= maxSoFar; num++)
                {
                    ans = (ans + dp(i + 1, maxSoFar, remain)) % MOD;
                }

                for (int num = maxSoFar + 1; num <= m; num++)
                {
                    ans = (ans + dp(i + 1, num, remain - 1)) % MOD;
                }

                memo[i][maxSoFar][remain] = ans;
                return ans;
            }

            /* 
            Approach 2: Bottom-Up Dynamic Programming
Complexity Analysis
•	Time complexity: O(n⋅m^2⋅k)
There are n⋅m⋅k possible states of dp. We iterate over each state in our nested for loops. To calculate a given state, we have for loops that iterate O(m) times. Thus, to calculate O(n⋅m⋅k) states costs O(n⋅m^2⋅k).
•	Space complexity: O(n⋅m⋅k)
Our dp table is of size O(n⋅m⋅k).

             */
            public int BottomUpDP(int n, int m, int k)
            {
                int[][][] dp = new int[n + 1][][];
                int MOD = (int)1e9 + 7;

                for (int num = 0; num < dp[0].Length; num++)
                {
                    dp[n][num][0] = 1;
                }

                for (int i = n - 1; i >= 0; i--)
                {
                    for (int maxSoFar = m; maxSoFar >= 0; maxSoFar--)
                    {
                        for (int remain = 0; remain <= k; remain++)
                        {
                            int ans = 0;
                            for (int num = 1; num <= maxSoFar; num++)
                            {
                                ans = (ans + dp[i + 1][maxSoFar][remain]) % MOD;
                            }

                            if (remain > 0)
                            {
                                for (int num = maxSoFar + 1; num <= m; num++)
                                {
                                    ans = (ans + dp[i + 1][num][remain - 1]) % MOD;
                                }
                            }

                            dp[i][maxSoFar][remain] = ans;
                        }
                    }
                }

                return dp[0][0][k];
            }
            /* 
                        Approach 3: Space-Optimized Dynamic Programming
                        Complexity Analysis
            •	Time complexity: O(n⋅m^2⋅k)
            There are n⋅m⋅k possible states of dp. We iterate over each state in our nested for loops. To calculate a given state, we have for loops that iterate O(m) times. Thus, to calculate O(n⋅m⋅k) states costs O(n⋅m^2⋅k).
            •	Space complexity: O(m⋅k)
            We have improved our space complexity by only requiring our tables to be of size O(m⋅k).

             */
            public int BottomUpDPWithSpaceOptimal(int n, int m, int k)
            {
                int[][] dp = new int[m + 1][];
                int[][] prevDp = new int[m + 1][];
                int MOD = (int)1e9 + 7;

                for (int num = 0; num < dp.Length; num++)
                {
                    prevDp[num][0] = 1;
                }

                for (int i = n - 1; i >= 0; i--)
                {
                    dp = new int[m + 1][];
                    for (int maxSoFar = m; maxSoFar >= 0; maxSoFar--)
                    {
                        for (int remain = 0; remain <= k; remain++)
                        {
                            int ans = 0;
                            for (int num = 1; num <= maxSoFar; num++)
                            {
                                ans = (ans + prevDp[maxSoFar][remain]) % MOD;
                            }

                            if (remain > 0)
                            {
                                for (int num = maxSoFar + 1; num <= m; num++)
                                {
                                    ans = (ans + prevDp[num][remain - 1]) % MOD;
                                }
                            }

                            dp[maxSoFar][remain] = ans;
                        }
                    }

                    prevDp = dp;
                }

                return dp[0][k];
            }

            /* Approach 4: A Different DP + Prefix Sums
            Complexity Analysis
            •	Time complexity: O(n⋅m⋅k)
            There are n⋅m⋅k possible states of dp. We iterate over each state in our nested for loops. Calculating a state now costs O(1), and we also maintain prefix while calculating the states of dp.
            •	Space complexity: O(n⋅m⋅k)
            Our dp and prefix tables are of size O(n⋅m⋅k).

             */
            public int DifferentDPWithPrefixSum(int n, int m, int k)
            {
                long[][][] dp = new long[n + 1][][];
                long[][][] prefix = new long[n + 1][][];
                int MOD = (int)1e9 + 7;

                for (int num = 1; num <= m; num++)
                {
                    dp[1][num][1] = 1;
                    prefix[1][num][1] = prefix[1][num - 1][1] + 1;
                }

                for (int i = 1; i <= n; i++)
                {
                    for (int maxNum = 1; maxNum <= m; maxNum++)
                    {
                        for (int cost = 1; cost <= k; cost++)
                        {
                            long ans = (maxNum * dp[i - 1][maxNum][cost]) % MOD;
                            ans = (ans + prefix[i - 1][maxNum - 1][cost - 1]) % MOD;

                            dp[i][maxNum][cost] += ans;
                            dp[i][maxNum][cost] %= MOD;

                            prefix[i][maxNum][cost] = (prefix[i][maxNum - 1][cost] + dp[i][maxNum][cost]);
                            prefix[i][maxNum][cost] %= MOD;
                        }
                    }
                }

                return (int)prefix[n][m][k];
            }
            /*             Approach 5: Space-Optimized Better DP
Complexity Analysis
•	Time complexity: O(n⋅m⋅k)
There are n⋅m⋅k possible states of dp. We iterate over each state in our nested for loops. Calculating a state now costs O(1), and we also maintain prefix while calculating the states of dp.
•	Space complexity: O(m⋅k)
Our dp and prefix tables are of size O(m⋅k).

             */
            public int BottomUpDPWithSpaceOptimalBetter(int n, int m, int k)
            {
                long[][] dp = new long[m + 1][];
                long[][] prefix = new long[m + 1][];
                long[][] prevDp = new long[m + 1][];
                long[][] prevPrefix = new long[m + 1][];
                int MOD = (int)1e9 + 7;

                for (int num = 1; num <= m; num++)
                {
                    dp[num][1] = 1;
                }

                for (int i = 1; i <= n; i++)
                {
                    if (i > 1)
                    {
                        dp = new long[m + 1][];
                    }

                    prefix = new long[m + 1][];

                    for (int maxNum = 1; maxNum <= m; maxNum++)
                    {
                        for (int cost = 1; cost <= k; cost++)
                        {
                            long ans = (maxNum * prevDp[maxNum][cost]) % MOD;
                            ans = (ans + prevPrefix[maxNum - 1][cost - 1]) % MOD;

                            dp[maxNum][cost] += ans;
                            dp[maxNum][cost] %= MOD;

                            prefix[maxNum][cost] = (prefix[maxNum - 1][cost] + dp[maxNum][cost]);
                            prefix[maxNum][cost] %= MOD;
                        }
                    }

                    prevDp = dp;
                    prevPrefix = prefix;
                }

                return (int)prefix[m][k];
            }
        }


        /* 1649. Create Sorted Array through Instructions
        https://leetcode.com/problems/create-sorted-array-through-instructions/description/
         */

        class CreateSortedArrayThruInstructionsSol
        {
            /* Approach 1: Segment Tree
            Complexity Analysis
Let N be the length of instructions and M be the maximum value in instructions.
•	Time Complexity: O(Nlog(M)). We need to iterate over instructions, and for each element, the time to find the left cost and right cost is O(log(M)), and we spend O(log(M)) inserting the current element into the Segment Tree. In total, we need O(N⋅log(M))=O(Nlog(M)).
•	Space Complexity: O(M), since we need an array of size 2M to store Segment Tree.

             */
            public int UsingSegmentTree(int[] instructions)
            {
                int m = (int)1e5 + 1;
                int[] tree = new int[m * 2];

                long cost = 0;
                long MOD = (int)1e9 + 7;
                foreach (int x in instructions)
                {
                    cost += Math.Min(Query(0, x, tree, m), Query(x + 1, m, tree, m));
                    Update(x, 1, tree, m);
                }
                return (int)(cost % MOD);
            }

            // implement Segment Tree
            private void Update(int index, int value, int[] tree, int m)
            {
                index += m;
                tree[index] += value;
                for (index >>= 1; index > 0; index >>= 1)
                    tree[index] = tree[index << 1] + tree[(index << 1) + 1];
            }

            private int Query(int left, int right, int[] tree, int m)
            {
                int result = 0;
                for (left += m, right += m; left < right; left >>= 1, right >>= 1)
                {
                    if ((left & 1) == 1)
                        result += tree[left++];
                    if ((right & 1) == 1)
                        result += tree[--right];
                }
                return result;
            }/* 
Approach 2: Binary Indexed Tree (BIT)
Complexity Analysis
Let N be the length of instructions and M be the maximum value in instructions.
•	Time Complexity: O(Nlog(M)). We need to iterate over instructions, and for each element, the time to find the left cost and right cost is O(log(M)), and we spend O(log(M)) inserting the current element into the BIT. In total, we need O(N⋅log(M))=O(Nlog(M)).
•	Space Complexity: O(M), since we need an array of size O(M) to store BIT.

 */
            public int UsingBIT(int[] instructions)
            {
                int m = 100002;
                int[] bit = new int[m];
                long cost = 0;
                long MOD = 1000000007;

                for (int i = 0; i < instructions.Length; i++)
                {
                    int leftCost = Query(instructions[i] - 1, bit);
                    int rightCost = i - Query(instructions[i], bit);
                    cost += Math.Min(leftCost, rightCost);
                    UpdateBIT(instructions[i], 1, bit, m);
                }
                return (int)(cost % MOD);
            }

            // implement Binary Index Tree
            private void UpdateBIT(int index, int value, int[] bit, int m)
            {
                index++;
                while (index < m)
                {
                    bit[index] += value;
                    index += index & -index;
                }
            }

            private int Query(int index, int[] bit)
            {
                index++;
                int result = 0;
                while (index >= 1)
                {
                    result += bit[index];
                    index -= index & -index;
                }
                return result;
            }
            /* Approach 3: Merge Sort

             */
            int[] smaller;
            int[] larger;
            int[][] temp; // record some temporal information

            public int UsingMergeSort(int[] instructions)
            {
                int n = instructions.Length;
                smaller = new int[n];
                larger = new int[n];
                temp = new int[n][];
                long cost = 0;
                long MOD = 1000000007;

                int[][] arrSmaller = new int[n][];
                int[][] arrLarger = new int[n][];
                for (int i = 0; i < n; i++)
                {
                    arrSmaller[i] = new int[] { instructions[i], i };
                    arrLarger[i] = new int[] { instructions[i], i };
                }

                SortSmaller(arrSmaller, 0, n - 1);
                SortLarger(arrLarger, 0, n - 1);

                for (int i = 0; i < n; i++)
                {
                    cost += Math.Min(smaller[i], larger[i]);
                }
                return (int)(cost % MOD);
            }

            private void SortSmaller(int[][] arr, int left, int right)
            {
                if (left == right)
                {
                    return;
                }
                int mid = (left + right) / 2;
                SortSmaller(arr, left, mid);
                SortSmaller(arr, mid + 1, right);
                MergeSmaller(arr, left, right, mid);
            }

            private void MergeSmaller(int[][] arr, int left, int right, int mid)
            {
                // merge [left, mid] and [mid+1, right]
                int i = left;
                int j = mid + 1;
                int k = left;
                // use temp[left...right] to temporarily store sorted array
                while (i <= mid && j <= right)
                {
                    if (arr[i][0] < arr[j][0])
                    {
                        temp[k++] = arr[i];
                        i++;
                    }
                    else
                    {
                        temp[k++] = arr[j];
                        smaller[arr[j][1]] += i - left;
                        j++;
                    }
                }
                while (i <= mid)
                {
                    temp[k++] = arr[i];
                    i++;
                }
                while (j <= right)
                {
                    temp[k++] = arr[j];
                    smaller[arr[j][1]] += i - left;
                    j++;
                }
                // restore from temp
                for (i = left; i <= right; i++)
                {
                    arr[i] = temp[i];
                }
            }

            private void SortLarger(int[][] arr, int left, int right)
            {
                if (left == right)
                {
                    return;
                }
                int mid = (left + right) / 2;
                SortLarger(arr, left, mid);
                SortLarger(arr, mid + 1, right);
                MergeLarger(arr, left, right, mid);
            }

            private void MergeLarger(int[][] arr, int left, int right, int mid)
            {
                // merge [left, mid] and [mid+1, right]
                int i = left;
                int j = mid + 1;
                int k = left;
                // use temp[left...right] to temporarily store sorted array
                while (i <= mid && j <= right)
                {
                    if (arr[i][0] <= arr[j][0])
                    {
                        temp[k++] = arr[i];
                        i++;
                    }
                    else
                    {
                        temp[k++] = arr[j];
                        larger[arr[j][1]] += mid - i + 1;
                        j++;
                    }
                }
                while (i <= mid)
                {
                    temp[k++] = arr[i];
                    i++;
                }
                while (j <= right)
                {
                    temp[k++] = arr[j];
                    larger[arr[j][1]] += mid - i + 1;
                    j++;
                }
                // restore from temp
                for (i = left; i <= right; i++)
                {
                    arr[i] = temp[i];
                }
            }
        }



        /* 154. Find Minimum in Rotated Sorted Array II
        https://leetcode.com/problems/find-minimum-in-rotated-sorted-array-ii/description/
         */
        public class FindMinInRotatedSortedArrayIISol
        {

            /* Approach 1: Variant of Binary Search
            Complexity Analysis
            •	Time complexity: on average O(log2N) where N is the length of the array, since in general it is a binary search algorithm. However, in the worst case where the array contains identical elements (i.e. case #3 nums[pivot]==nums[high]), the algorithm would deteriorate to iterating each element, as a result, the time complexity becomes O(N).
            •	Space complexity : O(1), it's a constant space solution.

             */
            public int UsingBinarySearchVariant(int[] nums)
            {
                int low = 0, high = nums.Length - 1;

                while (low < high)
                {
                    int pivot = low + (high - low) / 2;
                    if (nums[pivot] < nums[high])
                        high = pivot;
                    else if (nums[pivot] > nums[high])
                        low = pivot + 1;
                    else
                        high -= 1;
                }

                return nums[low];
            }
        }

        /* 
        425. Word Squares
        https://leetcode.com/problems/word-squares/description/
         */
        public class WordSquaresSol
        {
            private int numberOfLetters = 0;
            private string[] wordsArray = null;
            private Dictionary<string, List<string>> prefixHashTable = null;

            /* Approach 1: Backtracking with HashTable
            Complexity Analysis
            •	Time complexity: O(N⋅26^L), where N is the number of input words and L is the length of a single word.
            o	It is tricky to calculate the exact number of operations in the backtracking algorithm. We know that the trace of the backtrack would form a n-ary tree. Therefore, the upper bound of the operations would be the total number of nodes in a full-blossom n-ary tree.
            o	In our case, at any node of the trace, at maximum it could have 26 branches (i.e. 26 alphabet letters). Therefore, the maximum number of nodes in a 26-ary tree would be approximately 26^L.
            o	In the loop around the backtracking function, we enumerate the possibility of having each word as the starting word in the word square. As a result, in total the overall time complexity of the algorithm should be O(N⋅26^L).
            o	As large as the time complexity might appear, in reality, the actual trace of the backtracking is much smaller than its upper bound, thanks to the constraint checking (symmetric of word square) which greatly prunes the trace of the backtracking.
            •	Space Complexity: O(N⋅L+N⋅L/2)=O(N⋅L) where N is the number of words and L is the length of a single word.
            o	The first half of the space complexity (i.e. N⋅L) is the values in the hashtable, where we store L times all words in the hashtable.
            o	The second half of the space complexity (i.e. N⋅2L) is the space took by the keys of the hashtable, which include all prefixes of all words.
            o	In total, we could say that the overall space of the algorithm is proportional to the total words times the length of a single word.

             */
            public List<List<string>> BacktrackingWithDict(string[] words)
            {
                this.wordsArray = words;
                this.numberOfLetters = words[0].Length;

                List<List<string>> results = new List<List<string>>();
                this.BuildPrefixHashTable(words);

                foreach (string word in words)
                {
                    LinkedList<string> wordSquares = new LinkedList<string>();
                    wordSquares.AddLast(word);
                    this.Backtracking(1, wordSquares, results);
                }
                return results;
            }

            protected void Backtracking(int step, LinkedList<string> wordSquares,
                                        List<List<string>> results)
            {
                if (step == numberOfLetters)
                {
                    results.Add(new List<string>(wordSquares));
                    return;
                }

                StringBuilder prefix = new StringBuilder();
                foreach (string word in wordSquares)
                {
                    prefix.Append(word[step]);
                }

                foreach (string candidate in this.GetWordsWithPrefix(prefix.ToString()))
                {
                    wordSquares.AddLast(candidate);
                    this.Backtracking(step + 1, wordSquares, results);
                    wordSquares.RemoveLast();
                }
            }

            protected void BuildPrefixHashTable(string[] words)
            {
                this.prefixHashTable = new Dictionary<string, List<string>>();

                foreach (string word in words)
                {
                    for (int i = 1; i < this.numberOfLetters; ++i)
                    {
                        string prefix = word.Substring(0, i);
                        if (!this.prefixHashTable.TryGetValue(prefix, out List<string> wordList))
                        {
                            wordList = new List<string>();
                            wordList.Add(word);
                            this.prefixHashTable[prefix] = wordList;
                        }
                        else
                        {
                            wordList.Add(word);
                        }
                    }
                }
            }

            protected List<string> GetWordsWithPrefix(string prefix)
            {
                if (this.prefixHashTable.TryGetValue(prefix, out List<string> wordList))
                {
                    return wordList;
                }
                return new List<string>();
            }

            /* Approach 2: Backtracking with Trie
            Complexity Analysis
•	Time complexity: O(N⋅26^L⋅L), where N is the number of input words and L is the length of a single word.
o	Basically, the time complexity is same with the Approach #1 (O(N⋅26^L)), except that instead of the constant-time access for the function getWordsWithPrefix(prefix), we now need O(L).
•	Space Complexity: O(N⋅L+N⋅L/2)=O(N⋅L) where N is the number of words and L is the length of a single word.
o	The first half of the space complexity (i.e. N⋅L) is the word indice that we store in the Trie, where we store L times the index for each word.
o	The second half of the space complexity (i.e. N⋅L/2) is the space took by the prefixes of all words. In the worst case, we have no overlapping among the prefixes.
o	Overall, this approach has the same space complexity as the previous approach. Yet, in running time, it would consume less memory thanks to all the optimization that we have done.

             */
            private int N = 0;
            private string[] words = null;
            private TrieNode trie = null;

            public IList<IList<string>> BacktrackingWithTrie(string[] words)
            {
                this.words = words;
                this.N = words[0].Length;

                var results = new List<IList<string>>();
                this.BuildTrie(words);

                foreach (string word in words)
                {
                    var wordSquares = new LinkedList<string>();
                    wordSquares.AddLast(word);
                    this.Backtracking(1, wordSquares, results);
                }
                return results;


            }

            protected void Backtracking(int step, LinkedList<string> wordSquares, IList<IList<string>> results)
            {
                if (step == N)
                {
                    results.Add(new List<string>(wordSquares));
                    return;
                }

                var prefix = new StringBuilder();
                foreach (string word in wordSquares)
                {
                    prefix.Append(word[step]);
                }

                foreach (int wordIndex in GetWordsWithPrefix(prefix.ToString()))
                {
                    wordSquares.AddLast(this.words[wordIndex]);
                    this.Backtracking(step + 1, wordSquares, results);
                    wordSquares.RemoveLast();
                }
                List<int> GetWordsWithPrefix(string prefix)
                {
                    TrieNode node = this.trie;
                    foreach (char letter in prefix)
                    {
                        if (node.Children.ContainsKey(letter))
                        {
                            node = node.Children[letter];
                        }
                        else
                        {
                            // return an empty list.
                            return new List<int>();
                        }
                    }
                    return node.WordList;
                }

            }

            protected void BuildTrie(string[] words)
            {
                this.trie = new TrieNode();

                for (int wordIndex = 0; wordIndex < words.Length; ++wordIndex)
                {
                    string word = words[wordIndex];

                    TrieNode node = this.trie;
                    foreach (char letter in word)
                    {
                        if (node.Children.ContainsKey(letter))
                        {
                            node = node.Children[letter];
                        }
                        else
                        {
                            TrieNode newNode = new TrieNode();
                            node.Children[letter] = newNode;
                            node = newNode;
                        }
                        node.WordList.Add(wordIndex);
                    }
                }
            }


            public class TrieNode
            {
                public Dictionary<char, TrieNode> Children { get; } = new Dictionary<char, TrieNode>();
                public List<int> WordList { get; } = new List<int>();

                public TrieNode() { }
            }


        }

        /* 527. Word Abbreviation
        https://leetcode.com/problems/word-abbreviation/description/
         */
        class WordsAbbreviationSol
        {
            /* Approach #1: Greedy [Accepted]
            Complexity Analysis
•	Time Complexity: O(C^2) where C is the number of characters across all words in the given array.
•	Space Complexity: O(C)

             */
            public IList<string> UsingGreedy(IList<string> words)
            {
                int numberOfWords = words.Count;
                string[] abbreviations = new string[numberOfWords];
                int[] prefixCount = new int[numberOfWords];

                for (int index = 0; index < numberOfWords; ++index)
                    abbreviations[index] = Abbrev(words[index], 0);

                for (int index = 0; index < numberOfWords; ++index)
                {
                    while (true)
                    {
                        HashSet<int> duplicates = new HashSet<int>();
                        for (int j = index + 1; j < numberOfWords; ++j)
                            if (abbreviations[index].Equals(abbreviations[j]))
                                duplicates.Add(j);

                        if (duplicates.Count == 0) break;
                        duplicates.Add(index);
                        foreach (int k in duplicates)
                            abbreviations[k] = Abbrev(words[k], ++prefixCount[k]);
                    }
                }

                return new List<string>(abbreviations);
            }

            private string Abbrev(string word, int index)
            {
                int wordLength = word.Length;
                if (wordLength - index <= 3) return word;
                return word.Substring(0, index + 1) + (wordLength - index - 2) + word[wordLength - 1];
            }
            /* 
Approach #2: Group + Least Common Prefix [Accepted]
Complexity Analysis
•	Time Complexity: O(ClogC) where C is the number of characters across all words in the given array. The complexity is dominated by the sorting step.
•	Space Complexity: O(C).

 */
            public List<string> GroupWithLeastCommonPrefix(List<string> words)
            {
                Dictionary<string, List<IndexedWord>> groups = new Dictionary<string, List<IndexedWord>>();
                string[] result = new string[words.Count];

                int index = 0;
                foreach (string word in words)
                {
                    string abbreviation = Abbrev(word, 0);
                    if (!groups.ContainsKey(abbreviation))
                        groups[abbreviation] = new List<IndexedWord>();
                    groups[abbreviation].Add(new IndexedWord(word, index));
                    index++;
                }

                foreach (List<IndexedWord> group in groups.Values)
                {
                    group.Sort((a, b) => a.Word.CompareTo(b.Word));

                    int[] longestCommonPrefixArray = new int[group.Count];
                    for (int i = 1; i < group.Count; ++i)
                    {
                        int commonPrefixLength = LongestCommonPrefix(group[i - 1].Word, group[i].Word);
                        longestCommonPrefixArray[i] = commonPrefixLength;
                        longestCommonPrefixArray[i - 1] = Math.Max(longestCommonPrefixArray[i - 1], commonPrefixLength);
                    }

                    for (int i = 0; i < group.Count; ++i)
                        result[group[i].Index] = Abbrev(group[i].Word, longestCommonPrefixArray[i]);
                }

                return result.ToList();
            }


            public int LongestCommonPrefix(string word1, string word2)
            {
                int i = 0;
                while (i < word1.Length && i < word2.Length && word1[i] == word2[i])
                    i++;
                return i;
            }
            public class IndexedWord
            {
                public string Word { get; }
                public int Index { get; }

                public IndexedWord(string word, int index)
                {
                    Word = word;
                    Index = index;
                }
            }

            /* Approach #3: Group + Trie [Accepted
            Complexity Analysis
•	Time Complexity: O(C) where C is the number of characters across all words in the given array.
•	Space Complexity: O(C).

             */
            public IList<string> WordsAbbreviation(IList<string> words)
            {
                Dictionary<string, List<IndexedWord>> groups = new Dictionary<string, List<IndexedWord>>();
                string[] ans = new string[words.Count];

                int index = 0;
                foreach (string word in words)
                {
                    string ab = Abbrev(word, 0);
                    if (!groups.ContainsKey(ab))
                        groups[ab] = new List<IndexedWord>();
                    groups[ab].Add(new IndexedWord(word, index));
                    index++;
                }

                foreach (List<IndexedWord> group in groups.Values)
                {
                    TrieNode trie = new TrieNode();
                    foreach (IndexedWord iw in group)
                    {
                        TrieNode cur = trie;
                        foreach (char letter in iw.Word.Substring(1))
                        {
                            if (cur.Children[letter - 'a'] == null)
                                cur.Children[letter - 'a'] = new TrieNode();
                            cur.Count++;
                            cur = cur.Children[letter - 'a'];
                        }
                    }

                    foreach (IndexedWord iw in group)
                    {
                        TrieNode cur = trie;
                        int i = 1;
                        foreach (char letter in iw.Word.Substring(1))
                        {
                            if (cur.Count == 1) break;
                            cur = cur.Children[letter - 'a'];
                            i++;
                        }
                        ans[iw.Index] = Abbrev(iw.Word, i - 1);
                    }
                }

                return new List<string>(ans);
            }
            class TrieNode
            {
                public TrieNode[] Children { get; set; }
                public int Count { get; set; }
                public TrieNode()
                {
                    Children = new TrieNode[26];
                    Count = 0;
                }
            }

        }


        /* 548. Split Array with Equal Sum
        https://leetcode.com/problems/split-array-with-equal-sum/description/
         */

        public class SplitArrayWithEqualSumSol
        {
            /* Approach #1 Brute Force [Time Limit Exceeded]
Complexity Analysis
•	Time complexity : O(n^4). Four for loops inside each other each with a worst case run of length n.
•	Space complexity : O(1). Constant Space required.

             */
            public bool Naive(int[] nums)
            {
                if (nums.Length < 7)
                    return false;
                for (int i = 1; i < nums.Length - 5; i++)
                {
                    int sum1 = Sum(nums, 0, i);
                    for (int j = i + 2; j < nums.Length - 3; j++)
                    {
                        int sum2 = Sum(nums, i + 1, j);
                        for (int k = j + 2; k < nums.Length - 1; k++)
                        {
                            int sum3 = Sum(nums, j + 1, k);
                            int sum4 = Sum(nums, k + 1, nums.Length);
                            if (sum1 == sum2 && sum3 == sum4 && sum2 == sum4)
                                return true;
                        }
                    }
                }
                return false;
            }
            private int Sum(int[] nums, int l, int r)
            {
                int summ = 0;
                for (int i = l; i < r; i++)
                    summ += nums[i];
                return summ;
            }

            /* Approach #2 Cumulative Sum [Time Limit Exceeded]
            Complexity Analysis
•	Time complexity : O(n^3). Three for loops are there, one within the other.
•	Space complexity : O(n). sum array of size n is used for storing cumulative sum.

             */
            public bool CumulativeSum(int[] nums)
            {
                if (nums.Length < 7)
                    return false;
                int[] sum = new int[nums.Length];
                sum[0] = nums[0];
                for (int i = 1; i < nums.Length; i++)
                {
                    sum[i] = sum[i - 1] + nums[i];
                }
                for (int i = 1; i < nums.Length - 5; i++)
                {
                    int sum1 = sum[i - 1];
                    for (int j = i + 2; j < nums.Length - 3; j++)
                    {
                        int sum2 = sum[j - 1] - sum[i];
                        for (int k = j + 2; k < nums.Length - 1; k++)
                        {
                            int sum3 = sum[k - 1] - sum[j];
                            int sum4 = sum[nums.Length - 1] - sum[k];
                            if (sum1 == sum2 && sum3 == sum4 && sum2 == sum4)
                                return true;
                        }
                    }
                }
                return false;
            }

            /* Approach #3 Slightly Better Approach [Time Limit Exceeded]
            Complexity Analysis
•	Time complexity : O(n^3). Three loops are there.
•	Space complexity : O(n). sum array of size n is used for storing the cumulative sum.

             */
            public bool CumulativeSumSlightlyOptimal(int[] nums)
            {
                if (nums.Length < 7)
                    return false;

                int[] sum = new int[nums.Length];
                sum[0] = nums[0];
                for (int i = 1; i < nums.Length; i++)
                {
                    sum[i] = sum[i - 1] + nums[i];
                }
                for (int i = 1; i < nums.Length - 5; i++)
                {
                    int sum1 = sum[i - 1];
                    for (int j = i + 2; j < nums.Length - 3; j++)
                    {
                        int sum2 = sum[j - 1] - sum[i];
                        if (sum1 != sum2)
                            continue;
                        for (int k = j + 2; k < nums.Length - 1; k++)
                        {
                            int sum3 = sum[k - 1] - sum[j];
                            int sum4 = sum[nums.Length - 1] - sum[k];
                            if (sum3 == sum4 && sum2 == sum4)
                                return true;
                        }
                    }
                }
                return false;
            }
            /* Approach #4 Using HashMap [Time limit Exceeded] 
            Complexity Analysis
•	Time complexity : O(n^3). Three nested loops are there and every loop runs n times in the worst case. Consider the worstcase [0,0,0....,1,1,1,1,1,1,1].
•	Space complexity : O(n). HashMap size can go upto n.

            */
            public bool UsingDict(int[] nums)
            {
                Dictionary<int, List<int>> map = new Dictionary<int, List<int>>();
                int sum = 0, total = 0;

                for (int i = 0; i < nums.Length; i++)
                {
                    sum += nums[i];
                    if (map.ContainsKey(sum))
                        map[sum].Add(i);
                    else
                    {
                        map[sum] = new List<int>();
                        map[sum].Add(i);
                    }
                    total += nums[i];
                }

                sum = nums[0];
                for (int i = 1; i < nums.Length - 5; i++)
                {
                    if (map.ContainsKey(2 * sum + nums[i]))
                    {
                        for (int j = 0; j < map[2 * sum + nums[i]].Count; j++)
                        {
                            if (j > i + 1 && j < nums.Length - 3 && map.ContainsKey(3 * sum + nums[i] + nums[j]))
                            {
                                for (int k = 0; k < map[3 * sum + nums[j] + nums[i]].Count; k++)
                                {
                                    if (k < nums.Length - 1 && k > j + 1 && 4 * sum + nums[i] + nums[j] + nums[k] == total)
                                        return true;
                                }
                            }
                        }
                    }
                    sum += nums[i];
                }
                return false;
            }


            /* Approach #5 Using Cumulative Sum and HashSet [Accepted]
            Complexity Analysis
    •	Time complexity : O(n^2). One outer loop and two inner loops are used.
    •	Space complexity : O(n). HashSet size can go upto n.

             */
            public bool CumulativeSumWithHashSet(int[] nums)
            {
                if (nums.Length < 7)
                    return false;
                int[] sum = new int[nums.Length];
                sum[0] = nums[0];
                for (int i = 1; i < nums.Length; i++)
                {
                    sum[i] = sum[i - 1] + nums[i];
                }
                for (int j = 3; j < nums.Length - 3; j++)
                {
                    HashSet<int> set = new HashSet<int>();
                    for (int i = 1; i < j - 1; i++)
                    {
                        if (sum[i - 1] == sum[j - 1] - sum[i])
                            set.Add(sum[i - 1]);
                    }
                    for (int k = j + 2; k < nums.Length - 1; k++)
                    {
                        if (sum[nums.Length - 1] - sum[k] == sum[k - 1] - sum[j] && set.Contains(sum[k - 1] - sum[j]))
                            return true;
                    }
                }
                return false;
            }
        }


        /* 960. Delete Columns to Make Sorted III
        https://leetcode.com/problems/delete-columns-to-make-sorted-iii/description/
         */
        public class MinDeletionSizeSol
        {
            public int MinDeletionSize(IList<string> A)
            {
                int W = A[0].Length;
                int[] dp = new int[W];
                Array.Fill(dp, 1);
                for (int i = W - 2; i >= 0; i--)
                {
                    for (int j = i + 1; j < W; j++)
                    {
                        if (A.All(row => row[i] <= row[j]))
                        {
                            dp[i] = Math.Max(dp[i], 1 + dp[j]);
                        }
                    }
                }
                return W - dp.Max();
            }
        }

        /* 215. Kth Largest Element in an Array
        https://leetcode.com/problems/kth-largest-element-in-an-array/description/?envType=company&envId=facebook&favoriteSlug=facebook-all&difficulty=MEDIUM
         */

        class FindKthLargestSol
        {
            /*             Approach 1: Sort
            Complexity Analysis
            Given n as the length of nums,
            •	Time complexity: O(n⋅logn)
            Sorting nums requires O(n⋅logn) time.
            •	Space Complexity: O(logn) or O(n)
            The space complexity of the sorting algorithm depends on the implementation of each programming language:
            o	In Java, Arrays.sort() for primitives is implemented using a variant of the Quick Sort algorithm, which has a space complexity of O(logn)
            o	In C++, the sort() function provided by STL uses a hybrid of Quick Sort, Heap Sort, and Insertion Sort, with a worst-case space complexity of O(logn)
            o	In Python, the sort() function is implemented using the Timsort algorithm, which has a worst-case space complexity of O(n)

             */
            public int UsingSorting(int[] nums, int k)
            {
                Array.Sort(nums);
                // Can't sort int[] in descending order in Java;
                // Sort ascending and then return the kth element from the end
                return nums[nums.Length - k];
            }

            /*             Approach 2: Min-Heap
            Complexity Analysis
            Given n as the length of nums,
            •	Time complexity: O(n⋅logk)
            Operations on a heap cost logarithmic time relative to its size. Because our heap is limited to a size of k, operations cost at most O(logk). We iterate over nums, performing one or two heap operations at each iteration.
            We iterate n times, performing up to logk work at each iteration, giving us a time complexity of O(n⋅logk).
            Because k≤n, this is an improvement on the previous approach.
            •	Space complexity: O(k)
            The heap uses O(k) space.

             */
            public int UsingMinHeap(int[] nums, int k)
            {
                PriorityQueue<int, int> minHeap = new PriorityQueue<int, int>();
                foreach (int num in nums)
                {
                    minHeap.Enqueue(num, num);
                    if (minHeap.Count > k)
                    {
                        minHeap.Dequeue();
                    }
                }

                return minHeap.Peek();
            }

            /*             Approach 3: Quickselect
            Complexity Analysis
            Given n as the length of nums,
            •	Time complexity: O(n) on average, O(n^2) in the worst case
            Each call we make to quickSelect will cost O(n) since we need to iterate over nums to create left, mid, and right. The number of times we call quickSelect is dependent on how the pivots are chosen. The worst pivots to choose are the extreme (greatest/smallest) ones because they reduce our search space by the least amount. Because we are randomly generating pivots, we may end up calling quickSelect O(n) times, leading to a time complexity of O(n^2).
            However, the algorithm mathematically almost surely has a linear runtime. For any decent size of nums, the probability of the pivots being chosen in a way that we need to call quickSelect O(n) times is so low that we can ignore it.
            On average, the size of nums will decrease by a factor of ~2 on each call. You may think: that means we call quickSelect O(logn) times, wouldn't that give us a time complexity of O(n⋅logn)? Well, each successive call to quickSelect would also be on a nums that is a factor of ~2 smaller. This recurrence can be analyzed using the master theorem with a = 1, b = 2, k = 1:
            T(n)=T(n/2)+O(n)=O(n)
            •	Space complexity: O(n)
            We need O(n) space to create left, mid, and right. Other implementations of Quickselect can avoid creating these three in memory, but in the worst-case scenario, those implementations would still require O(n) space for the recursion call stack.

             */
            public int UsingQuickSelect(int[] numbers, int k)
            {
                List<int> numberList = new List<int>();
                foreach (int number in numbers)
                {
                    numberList.Add(number);
                }

                return QuickSelect(numberList, k);
            }

            private int QuickSelect(List<int> numbers, int k)
            {
                Random randomGenerator = new Random();
                int pivotIndex = randomGenerator.Next(numbers.Count);
                int pivot = numbers[pivotIndex];

                List<int> leftPartition = new List<int>();
                List<int> middlePartition = new List<int>();
                List<int> rightPartition = new List<int>();

                foreach (int number in numbers)
                {
                    if (number > pivot)
                    {
                        leftPartition.Add(number);
                    }
                    else if (number < pivot)
                    {
                        rightPartition.Add(number);
                    }
                    else
                    {
                        middlePartition.Add(number);
                    }
                }

                if (k <= leftPartition.Count)
                {
                    return QuickSelect(leftPartition, k);
                }

                if (leftPartition.Count + middlePartition.Count < k)
                {
                    return QuickSelect(rightPartition, k - leftPartition.Count - middlePartition.Count);
                }

                return pivot;
            }
            /*             Approach 4: Counting Sort
Complexity Analysis
Given n as the length of nums and m as maxValue - minValue,
•	Time complexity: O(n+m)
We first find maxValue and minValue, which costs O(n).
Next, we initialize count, which costs O(m).
Next, we populate count, which costs O(n).
Finally, we iterate over the indices of count, which costs up to O(m).
•	Space complexity: O(m)
We create an array count with size O(m).

             */
            public int UsingCountingSort(int[] nums, int k)
            {
                int minValue = int.MaxValue;
                int maxValue = int.MinValue;

                foreach (int num in nums)
                {
                    minValue = Math.Min(minValue, num);
                    maxValue = Math.Max(maxValue, num);
                }

                int[] count = new int[maxValue - minValue + 1];
                foreach (int num in nums)
                {
                    count[num - minValue]++;
                }

                int remain = k;
                for (int num = count.Length - 1; num >= 0; num--)
                {
                    remain -= count[num];
                    if (remain <= 0)
                    {
                        return num + minValue;
                    }
                }

                return -1;
            }
        }


        /* 
        162. Find Peak Element
        https://leetcode.com/problems/find-peak-element/description/?envType=company&envId=facebook&favoriteSlug=facebook-all&difficulty=MEDIUM
         */
        public class FindPeakElementSol
        {
            /*             Approach 1: Linear Scan
            Complexity Analysis
            •	Time complexity : O(n). We traverse the nums array of size n once only.
            •	Space complexity : O(1). Constant extra space is used.

             */
            public int LinearScan(int[] nums)
            {
                for (int i = 0; i < nums.Length - 1; i++)
                {
                    if (nums[i] > nums[i + 1])
                        return i;
                }

                return nums.Length - 1;
            }

            /* Approach 2: Recursive Binary Search
Complexity Analysis 
•	Time complexity : O(log2(n)). We reduce the search space in half at every step. Thus, the total search space will be consumed in log2(n) steps. Here, n refers to the size of nums array.
•	Space complexity : O(log2(n)). We reduce the search space in half at every step. Thus, the total search space will be consumed in log2(n) steps. Thus, the depth of recursion tree will go upto log2(n).

             */
            public int BinarySearchRec(int[] nums)
            {
                return Search(nums, 0, nums.Length - 1);
            }

            public int Search(int[] nums, int l, int r)
            {
                if (l == r)
                    return l;
                int mid = (l + r) / 2;
                if (nums[mid] > nums[mid + 1])
                    return Search(nums, l, mid);
                return Search(nums, mid + 1, r);
            }

            /*             Approach 3: Iterative Binary Search
Complexity Analysis
•	Time complexity : O(log2(n)). We reduce the search space in half at every step. Thus, the total search space will be consumed in log2(n) steps. Here, n refers to the size of nums array.
•	Space complexity : O(1). Constant extra space is used.

             */
            public int FindPeakElement(int[] nums)
            {
                int l = 0, r = nums.Length - 1;
                while (l < r)
                {
                    int mid = (l + r) / 2;
                    if (nums[mid] > nums[mid + 1])
                        r = mid;
                    else
                        l = mid + 1;
                }

                return l;
            }
        }

        /* 525. Contiguous Array
        https://leetcode.com/problems/contiguous-array/description/?envType=company&envId=facebook&favoriteSlug=facebook-all&difficulty=MEDIUM
         */

        public class FindMaxLengthSol
        {
            /* 
            Approach #1 Brute Force [Time Limit Exceeded]
            Complexity Analysis
            •	Time complexity : O(n^2). We consider every possible subarray by traversing over the complete array for every start point possible.
            •	Space complexity : O(1). Only two variables zeroes and ones are required.

             */
            public int Naive(int[] nums)
            {
                int maxlen = 0;
                for (int start = 0; start < nums.Length; start++)
                {
                    int zeroes = 0, ones = 0;
                    for (int end = start; end < nums.Length; end++)
                    {
                        if (nums[end] == 0)
                        {
                            zeroes++;
                        }
                        else
                        {
                            ones++;
                        }
                        if (zeroes == ones)
                        {
                            maxlen = Math.Max(maxlen, end - start + 1);
                        }
                    }
                }
                return maxlen;
            }
            /* Approach #2 Using Extra Array [Accepted]
            Complexity Analysis
•	Time complexity : O(n). The complete array is traversed only once.
•	Space complexity : O(n). arr array of size 2n+1 is used.

             */
            public int UsingExtraArray(int[] nums)
            {
                int[] arr = new int[2 * nums.Length + 1];
                Array.Fill(arr, -2);
                arr[nums.Length] = -1;
                int maxlen = 0, count = 0;
                for (int i = 0; i < nums.Length; i++)
                {
                    count = count + (nums[i] == 0 ? -1 : 1);
                    if (arr[count + nums.Length] >= -1)
                    {
                        maxlen = Math.Max(maxlen, i - arr[count + nums.Length]);
                    }
                    else
                    {
                        arr[count + nums.Length] = i;
                    }

                }
                return maxlen;
            }
            /* Approach #3 Using HashMap [Accepted]
            Complexity Analysis
•	Time complexity : O(n). The entire array is traversed only once.
•	Space complexity : O(n). Maximum size of the HashMap map will be n, if all the elements are either 1 or 0.

             */
            public int UsingHashMap(int[] nums)
            {
                Dictionary<int, int> map = new();
                map.Add(0, -1);
                int maxlen = 0, count = 0;
                for (int i = 0; i < nums.Length; i++)
                {
                    count = count + (nums[i] == 1 ? 1 : -1);
                    if (map.ContainsKey(count))
                    {
                        maxlen = Math.Max(maxlen, i - map[count]);
                    }
                    else
                    {
                        map[count] = i;
                    }
                }
                return maxlen;
            }
        }

        /* 
        238. Product of Array Except Self
        https://leetcode.com/problems/product-of-array-except-self/description/?envType=company&envId=facebook&favoriteSlug=facebook-all&difficulty=MEDIUM
         */

        class ProductOfArrayExceptSelfSol
        {
            /* 
            Approach 1: Left and Right product lists
            Complexity analysis
            •	Time complexity : O(N) where N represents the number of elements in the input array. We use one iteration to construct the array L, one to construct the array R and one last to construct the answer array using L and R.
            •	Space complexity : O(N) used up by the two intermediate arrays that we constructed to keep track of product of elements to the left and right.

             */
            public int[] UsingLeftandRightProductLists(int[] nums)
            {
                // The length of the input array
                int length = nums.Length;

                // The left and right arrays as described in the algorithm
                int[] L = new int[length];
                int[] R = new int[length];

                // Final answer array to be returned
                int[] answer = new int[length];

                // L[i] contains the product of all the elements to the left
                // Note: for the element at index '0', there are no elements to the left,
                // so L[0] would be 1
                L[0] = 1;
                for (int i = 1; i < length; i++)
                {
                    // L[i - 1] already contains the product of elements to the left of 'i - 1'
                    // Simply multiplying it with nums[i - 1] would give the product of all
                    // elements to the left of index 'i'
                    L[i] = nums[i - 1] * L[i - 1];
                }

                // R[i] contains the product of all the elements to the right
                // Note: for the element at index 'length - 1', there are no elements to the right,
                // so the R[length - 1] would be 1
                R[length - 1] = 1;
                for (int i = length - 2; i >= 0; i--)
                {
                    // R[i + 1] already contains the product of elements to the right of 'i + 1'
                    // Simply multiplying it with nums[i + 1] would give the product of all
                    // elements to the right of index 'i'
                    R[i] = nums[i + 1] * R[i + 1];
                }

                // Constructing the answer array
                for (int i = 0; i < length; i++)
                {
                    // For the first element, R[i] would be product except self
                    // For the last element of the array, product except self would be L[i]
                    // Else, multiple product of all elements to the left and to the right
                    answer[i] = L[i] * R[i];
                }

                return answer;
            }
            /* Approach 2: O(1) space approach
Complexity analysis
•	Time complexity : O(N) where N represents the number of elements in the input array. We use one iteration to construct the array L, one to update the array answer.
•	Space complexity : O(1) since don't use any additional array for our computations. The problem statement mentions that using the answer array doesn't add to the space complexity.

             */
            public int[] InConstantSpace(int[] nums)
            {
                // The length of the input array
                int length = nums.Length;

                // Final answer array to be returned
                int[] answer = new int[length];

                // answer[i] contains the product of all the elements to the left
                // Note: for the element at index '0', there are no elements to the left,
                // so the answer[0] would be 1
                answer[0] = 1;
                for (int i = 1; i < length; i++)
                {
                    // answer[i - 1] already contains the product of elements to the left of 'i - 1'
                    // Simply multiplying it with nums[i - 1] would give the product of all
                    // elements to the left of index 'i'
                    answer[i] = nums[i - 1] * answer[i - 1];
                }

                // R contains the product of all the elements to the right
                // Note: for the element at index 'length - 1', there are no elements to the right,
                // so the R would be 1
                int R = 1;
                for (int i = length - 1; i >= 0; i--)
                {
                    // For the index 'i', R would contain the
                    // product of all elements to the right. We update R accordingly
                    answer[i] = answer[i] * R;
                    R *= nums[i];
                }

                return answer;
            }

        }


        /* 1868. Product of Two Run-Length Encoded Arrays
        https://leetcode.com/problems/product-of-two-run-length-encoded-arrays/description/?envType=company&envId=facebook&favoriteSlug=facebook-all&difficulty=MEDIUM
         */
        class FindRLEArraySol
        {
            /* 
            Time and Space Complexity
Time Complexity
The time complexity of the given code primarily depends on these factors:
1.	Iterating through encoded1, which has n elements: O(n).
2.	A nested loop for processing elements in encoded2, which can be iterated up to m times in the worst case (in case the sizes in encoded1 are much larger): O(m).
However, since the second loop decreases the frequency from encoded2 without resetting, each element of encoded2 will be processed at most once throughout the entire iteration of encoded1. Therefore, the total time complexity will be O(n + m) where n is the total number of elements in encoded1 and m is the total number of elements in encoded2.
Space Complexity
The space complexity is determined by the size of the output, which in the worst case might contain an individual element for each multiplication of pairs from encoded1 and encoded2. In the worst case, every pair multiplication might result in a distinct value not matching the last element of the ans list, thus:
1.	The ans list: Up to O(n + m) in the worst case.
2.	Constant space for the pointers and temporary variables like vi, fi, f, v, j.
Therefore, the total space complexity of the given code would be O(n + m).

             */

            public List<List<int>> FindRLEArray(int[][] encoded1, int[][] encoded2)
            {
                // Initialize the answer list to hold the product RLE
                List<List<int>> result = new();
                // Index for tracking the current position in encoded2
                int currentIndex = 0;

                // Iterate over each pair in encoded1
                foreach (int[] pairEncoded1 in encoded1)
                {
                    // Grab the value and frequency from the current pair in encoded1
                    int value1 = pairEncoded1[0];
                    int frequency1 = pairEncoded1[1];

                    // Continue until we have processed all of this value
                    while (frequency1 > 0)
                    {
                        // Find the frequency to be processed which is the minimum of the
                        // remaining frequency from encoded1 and the current frequency from encoded2
                        int minFrequency = Math.Min(frequency1, encoded2[currentIndex][1]);
                        // Multiply the values from encoded1 and encoded2
                        int product = value1 * encoded2[currentIndex][0];

                        // Check if the last element in the result list has the same value
                        int resultSize = result.Count;
                        if (resultSize > 0 && result[resultSize - 1][0] == product)
                        {
                            // If yes, add the minFrequency to the frequency of the last element
                            int currentFreq = result[resultSize - 1][1];
                            result[resultSize - 1].Insert(1, currentFreq + minFrequency);
                        }
                        else
                        {
                            // If not, add a new pair with the product and minFrequency
                            result.Add(new List<int>(new List<int> { product, minFrequency }));
                        }
                        // Decrease the respective frequencies
                        frequency1 -= minFrequency;
                        encoded2[currentIndex][1] -= minFrequency;

                        // If we have processed all frequencies of the current pair in encoded2,
                        // move to the next pair
                        if (encoded2[currentIndex][1] == 0)
                        {
                            currentIndex++;
                        }
                    }
                }
                return result;
            }
        }

        /* 658. Find K Closest Elements
        https://leetcode.com/problems/find-k-closest-elements/description/?envType=company&envId=facebook&favoriteSlug=facebook-all&difficulty=MEDIUM
         */
        class FindKClosestElementsSol
        {
            /* Approach 1: Sort With Custom Comparator
            Complexity Analysis
Given N as the length of arr,
•	Time complexity: O(N⋅log(N)+k⋅log(k)).
To build sortedArr, we need to sort every element in the array by a new criteria: x - num. This costs O(N⋅log(N)). Then, we have to sort sortedArr again to get the output in ascending order. This costs O(k⋅log(k)) time since sortedArr.Length is only k.
•	Space complexity: O(N).
Before we slice sortedArr to contain only k elements, it contains every element from arr, which requires O(N) extra space. Note that we can use less space if we sort the input in place.

             */
            public List<int> FindClosestElements(int[] array, int count, int target)
            {
                // Convert from array to list first to make use of sorting
                List<int> sortedArray = new List<int>(array);

                // Sort using custom comparison
                sortedArray.Sort((num1, num2) => Math.Abs(num1 - target).CompareTo(Math.Abs(num2 - target)));

                // Only take 'count' elements
                sortedArray = sortedArray.GetRange(0, count);

                // Sort again to have output in ascending order
                sortedArray.Sort();
                return sortedArray;
            }

            /* Approach 2: Binary Search + Sliding Window
            Complexity Analysis
•	Time complexity: O(log(N)+k).
The initial binary search to find where we should start our window costs O(log(N)). Our sliding window initially starts with size 0 and we expand it one by one until it is of size k, thus it costs O(k) to expand the window.
•	Space complexity: O(1)
We only use integer variables left and right that are O(1) regardless of input size. Space used for the output is not counted towards the space complexity.

             */
            public List<int> UsingBinarySearchAndSlidingWindow(int[] arr, int k, int x)
            {
                List<int> result = new();

                // Base case
                if (arr.Length == k)
                {
                    for (int i = 0; i < k; i++)
                    {
                        result.Add(arr[i]);
                    }

                    return result;
                }

                // Binary search to find the closest element
                int left = 0;
                int right = arr.Length;
                int mid = 0;
                while (left < right)
                {
                    mid = (left + right) / 2;
                    if (arr[mid] >= x)
                    {
                        right = mid;
                    }
                    else
                    {
                        left = mid + 1;
                    }
                }

                // Initialize our sliding window's bounds
                left -= 1;
                right = left + 1;

                // While the window size is less than k
                while (right - left - 1 < k)
                {
                    // Be careful to not go out of bounds
                    if (left == -1)
                    {
                        right += 1;
                        continue;
                    }

                    // Expand the window towards the side with the closer number
                    // Be careful to not go out of bounds with the pointers
                    if (right == arr.Length || Math.Abs(arr[left] - x) <= Math.Abs(arr[right] - x))
                    {
                        left -= 1;
                    }
                    else
                    {
                        right += 1;
                    }
                }

                // Build and return the window
                for (int i = left + 1; i < right; i++)
                {
                    result.Add(arr[i]);
                }

                return result;
            }
            /* Approach 3: Binary Search To Find The Left Bound
            Complexity Analysis
Given N as the length of arr,
•	Time complexity: O(log(N−k)+k).
Although finding the bounds only takes O(log(N−k)) time from the binary search, it still costs us O(k) to build the final output.
Both the Java and Python implementations require O(k) time to build the result. However, it is worth noting that if the input array were given as a list instead of an array of integers, then the Java implementation could use the ArrayList.subList() method to build the result in O(1) time. If this were the case, the Java solution would have an (extremely fast) overall time complexity of O(log(N−k)).
•	Space complexity: O(1).
Again, we use a constant amount of space for our pointers, and space used for the output does not count towards the space complexity.

             */
            public List<int> UsingBinarySearchToFindTheLeftBound(int[] arr, int k, int x)
            {
                // Initialize binary search bounds
                int left = 0;
                int right = arr.Length - k;

                // Binary search against the criteria described
                while (left < right)
                {
                    int mid = (left + right) / 2;
                    if (x - arr[mid] > arr[mid + k] - x)
                    {
                        left = mid + 1;
                    }
                    else
                    {
                        right = mid;
                    }
                }

                // Create output in correct format
                List<int> result = new();
                for (int i = left; i < left + k; i++)
                {
                    result.Add(arr[i]);
                }

                return result;
            }
        }

        /* 494. Target Sum
        https://leetcode.com/problems/target-sum/description/?envType=company&envId=facebook&favoriteSlug=facebook-all&difficulty=MEDIUM
         */
        public class FindTargetSumWaysSol
        {
            int count = 0;
            /* 
            Approach 1: Brute Force
            Complexity Analysis
            •	Time complexity: O(2^n). Size of recursion tree will be 2^n. n refers to the size of nums array.
            •	Space complexity: O(n). The depth of the recursion tree can go up to n.

             */
            public int UsingRecursion(int[] nums, int S)
            {
                Calculate(nums, 0, 0, S);
                return count;
            }

            private void Calculate(int[] nums, int i, int sum, int S)
            {
                if (i == nums.Length)
                {
                    if (sum == S)
                    {
                        count++;
                    }
                }
                else
                {
                    Calculate(nums, i + 1, sum + nums[i], S);
                    Calculate(nums, i + 1, sum - nums[i], S);
                }
            }
            /* Approach 2: Recursion with Memoization 
            Complexity Analysis
•	Time complexity: O(t⋅n). The memo array of size O(t⋅n) has been filled just once. Here, t refers to the sum of the nums array and n refers to the length of the nums array.
•	Space complexity: O(t⋅n). The depth of recursion tree can go up to n.
The memo array contains t⋅n elements.

            */
            int total;

            public int UsingRecWithMemo(int[] nums, int S)
            {
                total = nums.Sum();

                int[][] memo = new int[nums.Length][];
                foreach (int[] row in memo)
                {
                    Array.Fill(row, int.MinValue);
                }
                return Calculate(nums, 0, 0, S, memo);
            }

            public int Calculate(int[] nums, int i, int sum, int S, int[][] memo)
            {
                if (i == nums.Length)
                {
                    if (sum == S)
                    {
                        return 1;
                    }
                    else
                    {
                        return 0;
                    }
                }
                else
                {
                    if (memo[i][sum + total] != int.MinValue)
                    {
                        return memo[i][sum + total];
                    }
                    int add = Calculate(nums, i + 1, sum + nums[i], S, memo);
                    int subtract = Calculate(nums, i + 1, sum - nums[i], S, memo);
                    memo[i][sum + total] = add + subtract;
                    return memo[i][sum + total];
                }
            }
            /*             Approach 3: 2D Dynamic Programming
            Complexity Analysis
•	Time complexity: O(t⋅n). The dp array of size O(t⋅n) has been filled just once. Here, t refers to the sum of the nums array and n refers to the length of the nums array.
•	Space complexity: O(t⋅n). dp array of size t⋅n is used.

             */
            public int BottomUp2DDP(int[] nums, int S)
            {
                int total = nums.Sum();
                int[][] dp = new int[nums.Length][];
                dp[0][nums[0] + total] = 1;
                dp[0][-nums[0] + total] += 1;

                for (int i = 1; i < nums.Length; i++)
                {
                    for (int sum = -total; sum <= total; sum++)
                    {
                        if (dp[i - 1][sum + total] > 0)
                        {
                            dp[i][sum + nums[i] + total] += dp[i - 1][sum + total];
                            dp[i][sum - nums[i] + total] += dp[i - 1][sum + total];
                        }
                    }
                }

                return Math.Abs(S) > total ? 0 : dp[nums.Length - 1][S + total];
            }
            /*             Approach 4: 1D Dynamic Programming
            Complexity Analysis
            •	Time complexity: O(t⋅n). Each of the n dp arrays of size t has been filled just once. Here, t refers to the sum of the nums array and n refers to the length of the nums array.
            •	Space complexity: O(t). Two dp arrays of size 2⋅t+1 are used, therefore the space usage is O(t).

             */
            public int BottomUp1DDP(int[] nums, int S)
            {
                int total = nums.Sum();
                int[] dp = new int[2 * total + 1];
                dp[nums[0] + total] = 1;
                dp[-nums[0] + total] += 1;

                for (int i = 1; i < nums.Length; i++)
                {
                    int[] next = new int[2 * total + 1];
                    for (int sum = -total; sum <= total; sum++)
                    {
                        if (dp[sum + total] > 0)
                        {
                            next[sum + nums[i] + total] += dp[sum + total];
                            next[sum - nums[i] + total] += dp[sum + total];
                        }
                    }
                    dp = next;
                }

                return Math.Abs(S) > total ? 0 : dp[S + total];
            }

        }

        /* 540. Single Element in a Sorted Array
        https://leetcode.com/problems/single-element-in-a-sorted-array/description/?envType=company&envId=facebook&favoriteSlug=facebook-all&difficulty=MEDIUM
         */

        class SingleNonDuplicateSol
        {
            /* Approach 1: Brute Force
            Complexity Analysis
•	Time complexity : O(n). For linear search, we are looking at every element in the array once.
•	Space complexity : O(1). We are only using constant extra space.
While this approach will work, the question tells us we need a O(logn) solution. Therefore, this solution isn't good enough.

             */
            public int Naive(int[] nums)
            {
                for (int i = 0; i < nums.Length - 1; i += 2)
                {
                    if (nums[i] != nums[i + 1])
                    {
                        return nums[i];
                    }
                }
                return nums[nums.Length - 1];
            }
            /* Approach 2: Binary Search
Complexity Analysis
•	Time complexity : O(logn). On each iteration of the loop, we're halving the number of items we still need to search.
•	Space complexity : O(1). We are only using constant space to keep track of where we are in the search.

             */
            public int UsingBinarySearch(int[] nums)
            {
                int lo = 0;
                int hi = nums.Length - 1;
                while (lo < hi)
                {
                    int mid = lo + (hi - lo) / 2;
                    bool halvesAreEven = (hi - mid) % 2 == 0;
                    if (nums[mid + 1] == nums[mid])
                    {
                        if (halvesAreEven)
                        {
                            lo = mid + 2;
                        }
                        else
                        {
                            hi = mid - 1;
                        }
                    }
                    else if (nums[mid - 1] == nums[mid])
                    {
                        if (halvesAreEven)
                        {
                            hi = mid - 2;
                        }
                        else
                        {
                            lo = mid + 1;
                        }
                    }
                    else
                    {
                        return nums[mid];
                    }
                }
                return nums[lo];
            }
            /*             Approach 3: Binary Search on Evens Indexes Only
            Complexity Analysis
            •	Time complexity : O(log2n)=O(logn). Same as the binary search above, except we are only binary searching half the elements, rather than all of them.
            •	Space complexity : O(1). Same as the other approaches. We are only using constant space to keep track of where we are in the search.

             */
            public int UsingBinarySearchOnEvenIndexesOnly(int[] nums)
            {
                int lo = 0;
                int hi = nums.Length - 1;
                while (lo < hi)
                {
                    int mid = lo + (hi - lo) / 2;
                    if (mid % 2 == 1) mid--;
                    if (nums[mid] == nums[mid + 1])
                    {
                        lo = mid + 2;
                    }
                    else
                    {
                        hi = mid;
                    }
                }
                return nums[lo];
            }
        }


        /* 1060. Missing Element in Sorted Array
        https://leetcode.com/problems/missing-element-in-sorted-array/description/?envType=company&envId=facebook&favoriteSlug=facebook-all&difficulty=MEDIUM
         */
        class MissingElementInSortedArraySol
        {

            /* Approach 1: Iteration
            Complexity Analysis
            Let n be the length of the input array nums.
            •	Time complexity: O(n)
            o	The algorithm requires iterating over the entire array at most once to calculate the number of missing elements between each adjacent pair of numbers.
            •	Space complexity: O(1)
            o	We only use a constant amount of extra space for variables missed_in_gap and i.

             */
            public int UsingIterative(int[] nums, int k)
            {
                int n = nums.Length;

                for (int i = 1; i < n; ++i)
                {
                    int missedInGap = nums[i] - nums[i - 1] - 1;
                    if (missedInGap >= k)
                    {
                        return nums[i - 1] + k;
                    }
                    k -= missedInGap;
                }

                return nums[n - 1] + k;
            }
            /* Approach 2: Binary Search
            Complexity Analysis
            LLet n be the length of the input array nums.
            •	Time complexity: O(logn)
            o	The algorithm repeatedly divides the search space in half using binary search until it finds the kth missing element.
            •	Space complexity: O(1)
            o	The algorithm only uses a constant amount of extra space for variables left, right, and mid.

             */
            public int UsingBinarySearch(int[] nums, int k)
            {
                int n = nums.Length;
                int left = 0, right = n - 1;

                while (left < right)
                {
                    int mid = right - (right - left) / 2;
                    if (nums[mid] - nums[0] - mid < k)
                    {
                        left = mid;
                    }
                    else
                    {
                        right = mid - 1;
                    }
                }

                return nums[0] + k + left;
            }

        }


        /* 1043. Partition Array for Maximum Sum
        https://leetcode.com/problems/partition-array-for-maximum-sum/description/?envType=company&envId=facebook&favoriteSlug=facebook-all&difficulty=MEDIUM
         */
        class MaxSumAfterPartitioningSol
        {
            /* Approach 1: Top-Down Dynamic Programming
            Complexity Analysis
Let N be the number of elements in the array, and K is the maximum length of a subarray.
•	Time complexity: O(N⋅K)
The time complexity for the recursion with memoization is equal to the number of times maxSum() is called times the average time of maxSum(). The number of calls to maxSum() is N because each non-memoized call will call maxSum() only once. Each memoized call will take O(1) time, while for the non-memoized call, we iterate over most K elements ahead of it. Hence, the total time complexity equals O(N⋅K).
•	Space complexity: O(N)
The result for each start index will be stored in dp, and start can have the values from 0 to N; thus, the space required is O(N). Also, the stack space needed for recursion is equal to the maximum number of active function calls which will be N, one for each index. Hence, the space complexity will equal O(N).

             */
            public int TopDownDP(int[] arr, int k)
            {
                int[] dp = new int[arr.Length];
                Array.Fill(dp, -1);

                return MaxSum(arr, k, dp, 0);
            }
            private int MaxSum(int[] arr, int k, int[] dp, int start)
            {
                int N = arr.Length;

                if (start >= N)
                {
                    return 0;
                }

                // Return the already calculated answer.
                if (dp[start] != -1)
                {
                    return dp[start];
                }

                int currMax = 0, ans = 0;
                int end = Math.Min(N, start + k);
                for (int i = start; i < end; i++)
                {
                    currMax = Math.Max(currMax, arr[i]);
                    // Store the maximum of all options for the current subarray.
                    ans = Math.Max(ans, currMax * (i - start + 1) + MaxSum(arr, k, dp, i + 1));
                }

                // Store the answer to be reused.
                return dp[start] = ans;
            }
            /* Approach 2: Bottom-Up Dynamic Programming
Complexity Analysis
Let N be the number of elements in the array, and K is the maximum length of a subarray.
•	Time complexity: O(N⋅K)
We iterate over all elements from N - 1 to 0, and, for each of them, iterate over at most K elements so the time complexity is equal to O(N⋅K).
•	Space complexity: O(N)
The result for each start index will be stored in dp, and start can have the values from 0 to N; thus, the space required is O(N).


             */
            public int BottomUpDP(int[] arr, int k)
            {
                int N = arr.Length;

                int[] dp = new int[N + 1];
                Array.Fill(dp, 0);

                for (int start = N - 1; start >= 0; start--)
                {
                    int currMax = 0;
                    int end = Math.Min(N, start + k);

                    for (int i = start; i < end; i++)
                    {
                        currMax = Math.Max(currMax, arr[i]);
                        // Store the maximum of all options for the current subarray.
                        dp[start] = Math.Max(dp[start], dp[i + 1] + currMax * (i - start + 1));
                    }
                }
                return dp[0];
            }

            /* Note: If we observe closely the above approach we need only the last K previously calculated answers to find the 
            answer for the current index start. This is because the length of the subarray cannot be more than K, even though we have stored all N answers in the array dp. We can change the size of array dp to store only the last K results and then use them to calculate the answers. This will reduce the space complexity to O(K), which is a significant reduction when K is small. At worst, when K is equal to N, the space complexity will still be O(N). The time complexity of this approach is O(N⋅K). The following code is written using this approach: */
            public int BottomUpDPWithKElementStoreOnly(int[] arr, int k)
            {
                int N = arr.Length;
                int K = k + 1;

                int[] dp = new int[K];
                Array.Fill(dp, 0);

                for (int start = N - 1; start >= 0; start--)
                {
                    int currMax = 0;
                    int end = Math.Min(N, start + k);

                    for (int i = start; i < end; i++)
                    {
                        currMax = Math.Max(currMax, arr[i]);
                        dp[start % K] = Math.Max(dp[start % K], dp[(i + 1) % K] + currMax * (i - start + 1));
                    }
                }
                return dp[0];
            }


        }

        /*  287. Find the Duplicate Number
        https://leetcode.com/problems/find-the-duplicate-number/description/?envType=company&envId=facebook&favoriteSlug=facebook-all&difficulty=MEDIUM
         */
        class FindDuplicateSol
        {

            /* Approach 1: Sort
            Complexity Analysis
            •	Time Complexity: O(nlogn)
            Sorting takes O(nlogn) time. This is followed by a linear scan, resulting in a total of O(nlogn) + O(n) = O(nlogn) time.
            •	Space Complexity: O(logn) or O(n)
            The space complexity of the sorting algorithm depends on the implementation of each programming language:
            o	In Java, Arrays.sort() for primitives is implemented using a variant of the Quick Sort algorithm, which has a space complexity of O(logn)
            o	In C++, the sort() function provided by STL uses a hybrid of Quick Sort, Heap Sort and Insertion Sort, with a worst case space complexity of O(logn)
            o	In Python, the sort() function is implemented using the Timsort algorithm, which has a worst-case space complexity of O(n)

             */
            public int UsingSort(int[] nums)
            {
                Array.Sort(nums);
                for (int i = 1; i < nums.Length; i++)
                {
                    if (nums[i] == nums[i - 1])
                        return nums[i];
                }

                return -1;
            }

            /* Approach 2: Set
            Note: This approach does not use constant space, and hence does not meet the problem constraints. However, it utilizes a fundamental concept that can help solve similar problems.


            Complexity Analysis
•	Time Complexity: O(n)
HashSet insertions and lookups have amortized constant time complexities. Hence, this algorithm requires linear time, since it consists of a single for loop that iterates over each element, looking up the element and inserting it into the set at most once.
•	Space Complexity: O(n)
We use a set that may need to store at most n elements, leading to a linear space complexity of O(n).

             */
            public int UsingHashSet(int[] nums)
            {
                HashSet<int> seen = new();
                foreach (int num in nums)
                {
                    if (seen.Contains(num))
                        return num;
                    seen.Add(num);
                }
                return -1;
            }
            /* Approach 3: Negative Marking
Complexity Analysis
•	Time Complexity: O(n)
Each element is visited at most twice (once in the first loop to find the duplicate and once in the second loop to restore the numbers).
•	Space Complexity: O(1)
All manipulation is done in place, so no additional storage (barring one variable) is needed.

             */
            public int UsingNegativeMarking(int[] nums)
            {
                int duplicate = -1;
                for (int i = 0; i < nums.Length; i++)
                {
                    int cur = Math.Abs(nums[i]);
                    if (nums[cur] < 0)
                    {
                        duplicate = cur;
                        break;
                    }
                    nums[cur] *= -1;
                }

                // Restore numbers
                for (int i = 0; i < nums.Length; i++)
                    nums[i] = Math.Abs(nums[i]);

                return duplicate;
            }
            /* Approach 4.1: Array as HashMap (Recursion)
            Complexity Analysis
•	Time Complexity: O(n)
The function (store) makes a single recursive call at every instance. Each index is visited at most once, resulting in O(n) time complexity.
•	Space Complexity: O(n)
Since this is a recursive function with up to n self-invocations (i.e. depth of the recursive function = n), the space complexity is O(n) as there can be up to n function calls in memory at some point. Note that due to the recursive nature of the solution, there is some additional overhead on each invocation (such as the function variables and a return function pointer that are stored on the system executable stack).

             */
            public int UsingArrayAsDictRec(int[] nums)
            {
                return Store(nums, 0);
            }
            public int Store(int[] nums, int cur)
            {
                if (cur == nums[cur])
                    return cur;
                int nxt = nums[cur];
                nums[cur] = cur;
                return Store(nums, nxt);
            }
            /*             Approach 4.2: Array as HashMap (Iterative)
Complexity Analysis
•	Time Complexity: O(n)
Each number needs to be swapped at most once before it is placed in its desired position.
•	Space Complexity: O(1)
No additional storage is needed (barring the temporary variables used for swapping).

             */
            public int UsingArrayAsDictIterative(int[] nums)
            {
                while (nums[0] != nums[nums[0]])
                {
                    int nxt = nums[nums[0]];
                    nums[nums[0]] = nums[0];
                    nums[0] = nxt;
                }
                return nums[0];
            }

            /* Approach 5: Binary Search
            Complexity Analysis
            •	Time Complexity: O(nlogn)
            The outer loop uses binary search to identify a candidate - this runs in O(logn) time. For each candidate, we iterate over the entire array which takes O(n) time, resulting in a total of O(nlogn) time.
            •	Space Complexity: O(1)
            No additional storage is needed (barring a few variables), resulting in a constant O(1) space complexity.

             */
            public int UsingBinarySearch(int[] nums)
            {
                // 'low' and 'high' represent the range of values of the target        
                int low = 1, high = nums.Length - 1;
                int duplicate = -1;

                while (low <= high)
                {
                    int cur = (low + high) / 2;

                    // Count how many numbers in 'nums' are less than or equal to 'cur'
                    int count = 0;
                    foreach (int num in nums)
                    {
                        if (num <= cur)
                            count++;
                    }

                    if (count > cur)
                    {
                        duplicate = cur;
                        high = cur - 1;
                    }
                    else
                    {
                        low = cur + 1;
                    }
                }
                return duplicate;
            }
            /* Approach 6: Sum of Set Bits
            Complexity Analysis
            Let n be the length of nums and m be the bit-length of n.
            •	Time Complexity: O(nlogn)
            The outer loop runs a maximum of m times (once for each bit in n). The inner loop goes over all n elements in nums, resulting in a total time complexity of O(m⋅n).
            It is a common misconception to treat m as a constant because it is small and thus consider this to be a linear time complexity algorithm. Setting the problem constraints aside, the value of m depends on n. More specifically, m is the bit-length of n which is approximately equal to log2(n). Thus this algorithm has linearithmic time complexity.
            •	Space Complexity: O(1)
            No additional storage is needed (barring a few variables), resulting in a constant O(1) space complexity.

             */
            public int UsingSumOfSetBits(int[] nums)
            {
                int duplicate = 0;
                int n = nums.Length - 1;
                int max_num = FindMaxNum(nums);
                int max_bit = CalcMaxBit(max_num);

                // Iterate over each bit        
                for (int bit = 0; bit < max_bit; bit++)
                {
                    int mask = (1 << bit);
                    int base_count = 0, nums_count = 0;

                    for (int i = 0; i <= n; i++)
                    {
                        // If bit is set in number (i) then add 1 to base_count                
                        if ((i & mask) > 0)
                        {
                            base_count++;
                        }
                        // If bit is set in nums[i] then add 1 to nums_count
                        if ((nums[i] & mask) > 0)
                        {
                            nums_count++;
                        }
                    }

                    // If the current bit is more frequently set in nums than it is in 
                    // the range [1, 2, ..., n] then it must also be set in the duplicate number
                    if (nums_count > base_count)
                    {
                        duplicate |= mask;
                    }
                }
                return duplicate;
            }
            // Find the position of the Most Significant Bit in num    
            int CalcMaxBit(int num)
            {
                int bits = 0;
                while (num > 0)
                {
                    num /= 2;
                    bits++;
                }
                return bits;
            }

            // Find the largest number in nums
            int FindMaxNum(int[] nums)
            {
                int max_num = 0;
                for (int i = 0; i < nums.Length; i++)
                    max_num = Math.Max(max_num, nums[i]);
                return max_num;
            }

            /* Approach 7: Floyd's Tortoise and Hare (Cycle Detection)
            Complexity Analysis
•	Time Complexity: O(n)
•	Space Complexity: O(1)

             */
            public int UsingFloydsTortoiseAndHare(int[] nums)
            {

                // Find the intersection point of the two runners.
                int tortoise = nums[0];
                int hare = nums[0];

                do
                {
                    tortoise = nums[tortoise];
                    hare = nums[nums[hare]];
                } while (tortoise != hare);

                // Find the "entrance" to the cycle.
                tortoise = nums[0];

                while (tortoise != hare)
                {
                    tortoise = nums[tortoise];
                    hare = nums[hare];
                }

                return hare;
            }

        }

        /* 1509. Minimum Difference Between Largest and Smallest Value in Three Moves
        https://leetcode.com/problems/minimum-difference-between-largest-and-smallest-value-in-three-moves/description/
         */
        class MinDifferenceBetweenLargestAndSmallestValueInThreeMovesSol
        {

            /* Approach 1: Sort + Greedy Deletion
            Complexity Analysis
            Let n be the length of the array nums.
            •	Time Complexity: O(n⋅logn)
            Sorting the array nums takes O(nlogn) time. The for loop runs a fixed number of 4 iterations, taking O(1) time. Thus, the overall time complexity is O(nlogn).
            •	Space Complexity: O(n) or O(logn)
            When we sort the nums array, some extra space is used. The space complexity of the sorting algorithm depends on the programming language.
            In Python, the sort method sorts a list using the Timsort algorithm, which combines Merge Sort and Insertion Sort and has O(n) additional space.
            In Java, Arrays.sort() is implemented using a variant of the Quick Sort algorithm, with a space complexity of O(logn) for sorting two arrays.
            In C++, the sort() function is implemented as a hybrid of Quick Sort, Heap Sort, and Insertion Sort, with a worse-case space complexity of O(logn).

             */
            public int UsingSortAndGreedyDeletion(int[] nums)
            {
                int numsSize = nums.Length;

                // If the array has 4 or fewer elements, return 0
                if (numsSize <= 4) return 0;

                Array.Sort(nums);

                int minDiff = int.MaxValue;

                // Four scenarios to compute the minimum difference
                for (int left = 0, right = numsSize - 4; left < 4; left++, right++)
                {
                    minDiff = Math.Min(minDiff, nums[right] - nums[left]);
                }

                return minDiff;
            }
            /* Approach 2: Partial Sort + Greedy Deletion	
Complexity Analysis
Let n be the length of the array nums.
•	Time Complexity: O(n)
o	Java and Python:
o	Finding the 4 smallest elements using a max heap takes O(n) time because maintaining a fixed-size heap of 4 elements results in O(n⋅log4)=O(n).
o	Sorting the 4 smallest elements takes O(1) time because sorting a constant number of elements is constant time.
o	Finding the 4 largest elements using a min heap also takes O(n) time because maintaining a fixed-size heap of 4 elements results in O(n⋅log4)=O(n).
o	Sorting the 4 largest elements takes O(1) time.
o	C++:
o	The partial_sort function runs in O(n⋅log4)=O(n) time as it sorts only the first four elements and ensures the smallest four elements are in place.
o	The nth_element function, which partitions the array around the 4th largest element, also runs in O(n) time.
o	The sort function, which sorts the last four elements, runs in O(4⋅log4)=O(1) time because sorting a constant number of elements is constant time.
The for loop that runs a fixed number of 4 iterations takes O(1) time.
Therefore, the total time complexity is O(n).
•	Space Complexity: O(1)
o	Java and Python: The algorithm uses constant space to store the heaps and intermediate results, which do not grow with the input size. This includes space for the heaps (each with a maximum size of 4) and any additional variables.
o	C++: The algorithm uses constant space regardless of the input size, as it only requires a few variables for indexing and storing intermediate results.
Therefore, the total space complexity is O(1).

             */
            public int UsingPartialSortAndGreedyDeletion(int[] nums)
            {
                int numsSize = nums.Length;
                if (numsSize <= 4)
                {
                    return 0;
                }

                // Find the four smallest elements using a fixed-size max 
                //TODO:Test below MaxHeap Comparision
                PriorityQueue<int, int> maxHeap = new PriorityQueue<int, int>(Comparer<int>.Create((a, b) => b.CompareTo(a)));
                foreach (int num in nums)
                {
                    maxHeap.Enqueue(num, num);
                    if (maxHeap.Count > 4)
                    {
                        maxHeap.Dequeue();
                    }
                }
                List<int> smallestFour = new List<int>();
                while (maxHeap.Count > 0)
                {
                    smallestFour.Add(maxHeap.Dequeue());
                }

                smallestFour.Sort();

                // Find the four largest elements using a fixed-size min heap
                PriorityQueue<int, int> minHeap = new PriorityQueue<int, int>();
                foreach (int num in nums)
                {
                    minHeap.Enqueue(num, num);
                    if (minHeap.Count > 4)
                    {
                        minHeap.Dequeue();
                    }
                }
                List<int> largestFour = new List<int>();
                while (minHeap.Count > 0)
                {
                    largestFour.Add(minHeap.Dequeue());
                }
                largestFour.Sort();

                int minDiff = int.MaxValue;
                // Four scenarios to compute the minimum difference
                for (int i = 0; i < 4; i++)
                {
                    minDiff = Math.Min(
                        minDiff,
                        largestFour[i] - smallestFour[i]
                    );
                }

                return minDiff;
            }



        }

        /* 945. Minimum Increment to Make Array Unique
        https://leetcode.com/problems/minimum-increment-to-make-array-unique/description/
         */

        class MinIncrementToMakeArrayUniqueSol
        {

            /* Approach 1: Sorting
            Complexity Analysis
            Let n be the length of the array nums.
            •	Time complexity: O(n⋅logn)
            Sorting the array requires O(n⋅logn) time and a single traversal over the entire array takes O(n) time. This leads to an overall time complexity of O(n⋅logn)+O(n), which simplifies to a O(n⋅logn) time complexity.
            •	Space complexity: O(n) or O(logn)
            Sorting arrays in place requires some additional space. The space complexity of sorting algorithms varies depending on the programming language being used:
            o	Python's sort method employs the Tim Sort algorithm, which is a combination of Merge Sort and Insertion Sort. This algorithm has a space complexity of O(n).
            o	In C++, the sort() function is a hybrid implementation that incorporates Quick Sort, Heap Sort, and Insertion Sort. Its worst-case space complexity is O(log n).
            o	Java's Arrays.sort() method uses a variation of the Quick Sort algorithm. When sorting two arrays, it has a space complexity of O(log n).

             */
            public int UsingSort(int[] nums)
            {
                int minIncrements = 0;

                Array.Sort(nums);

                for (int i = 1; i < nums.Length; i++)
                {
                    // Ensure each element is greater than its previous
                    if (nums[i] <= nums[i - 1])
                    {
                        // Add the required increment to minIncrements
                        int increment = nums[i - 1] + 1 - nums[i];
                        minIncrements += increment;

                        // Set the element to be greater than its previous
                        nums[i] = nums[i - 1] + 1;
                    }
                }

                return minIncrements;
            }
            /* Approach 2: Counting
Complexity Analysis
Let n be the length of nums and max be the maximum element in nums.
•	Time complexity: O(n+max)
The algorithm initially iterates over nums twice, each iteration taking O(n) time. To find the number of increments, it then loops over the frequencyCount array, which has a time complexity of O(n+max). Thus, the total time complexity is 2⋅O(n)+O(n+max), simplifying to O(n+max).
•	Space complexity: O(n+max)
The only additional space used by the algorithm is the frequencyCount array, which has a size of n+max. Therefore, the space complexity is O(n+max).

             */
            public int UsingCounting(int[] nums)
            {
                int n = nums.Length;
                int max = 0;
                int minIncrements = 0;

                // Find maximum value in array to determine range of frequencyCount array
                foreach (int val in nums)
                {
                    max = Math.Max(max, val);
                }

                // Create a frequencyCount array to store the frequency of each value in nums
                int[] frequencyCount = new int[n + max];

                // Populate frequencyCount array with the frequency of each value in nums
                foreach (int val in nums)
                {
                    frequencyCount[val]++;
                }

                // Iterate over the frequencyCount array to make all values unique
                for (int i = 0; i < frequencyCount.Length; i++)
                {
                    if (frequencyCount[i] <= 1) continue;

                    // Determine excess occurrences, carry them over to the next value,
                    // ensure single occurrence for current value, and update minIncrements.
                    int duplicates = frequencyCount[i] - 1;
                    frequencyCount[i + 1] += duplicates;
                    frequencyCount[i] = 1;
                    minIncrements += duplicates;
                }

                return minIncrements;
            }

        }

        /* 1151. Minimum Swaps to Group All 1's Together
        https://leetcode.com/problems/minimum-swaps-to-group-all-1s-together/description/
         */
        class MinimumSwapsToGroupAll1sTogetherISol
        {
            /*             Approach 1: Sliding Window with Two Pointers
            Complexity Analysis
•	Time complexity: O(n), when n is the length of the array.
•	Space complexity: O(1).

             */
            public int UsingSlidingWindowWithTwoPointers(int[] data)
            {
                int ones = data.Sum();
                int cnt_one = 0, max_one = 0;
                int left = 0, right = 0;

                while (right < data.Length)
                {
                    // updating the number of 1's by adding the new element
                    cnt_one += data[right++];
                    // maintain the length of the window to ones
                    if (right - left > ones)
                    {
                        // updating the number of 1's by removing the oldest element
                        cnt_one -= data[left++];
                    }
                    // record the maximum number of 1's in the window
                    max_one = Math.Max(max_one, cnt_one);
                }
                return ones - max_one;
            }

            /* Approach 2: Sliding Window with Deque (Double-ended Queue) 
            Complexity Analysis
•	Time complexity: O(n), when n is the length of the array.
Note that the time complexities of adding the element to deque's right end and popping the element from its left end are both O(1).
•	Space complexity: O(n), when n is the length of the array. This is because we need a Deque of size ones.

            */
            public int UsingSlidingWindowWithDeque(int[] data)
            {
                int ones = data.Sum();
                int cnt_one = 0, max_one = 0;
                // maintain a deque with the size = ones
                LinkedList<int> deque = new();

                for (int i = 0; i < data.Length; i++)
                {

                    // we would always add the new element into the deque
                    deque.AddLast(data[i]);
                    cnt_one += data[i];

                    // when there are more than ones elements in the deque,
                    // remove the leftmost one
                    if (deque.Count > ones)
                    {
                        cnt_one -= deque.First();
                        deque.RemoveFirst();
                    }
                    max_one = Math.Max(max_one, cnt_one);
                }
                return ones - max_one;

            }
        }

        /* 2134. Minimum Swaps to Group All 1's Together II
        https://leetcode.com/problems/minimum-swaps-to-group-all-1s-together-ii/description/
         */

        class MinimumSwapsToGroupAll1sTogetherIISol
        {


            /* Approach 1: Using Suffix Sum
            Complexity Analysis
            Let n be the length of the array.
            •	Time complexity: O(n)
            The main operations (constructing the rightSuffixSum and calculating minimum swaps) involve single passes through the array.
            •	Space complexity: O(n)
            O(n), due to the rightSuffixSum array, which stores the count for each position in the input array.

             */
            public int UsingSuffixSum(int[] nums)
            {
                // Calculate the minimum swaps needed to group all 1s or all 0s together
                int op1 = MinSwapsHelper(nums, 0); // Grouping all 0s together
                int op2 = MinSwapsHelper(nums, 1); // Grouping all 1s together
                return Math.Min(op1, op2);
            }

            // Helper function to calculate the minimum swaps required to group all `val` together
            private int MinSwapsHelper(int[] data, int val)
            {
                int length = data.Length;
                int[] rightSuffixSum = new int[length + 1];

                // Construct the suffix sum array for counting opposite values (val ^ 1)
                for (int i = length - 1; i >= 0; i--)
                {
                    rightSuffixSum[i] = rightSuffixSum[i + 1];
                    if (data[i] == (val ^ 1)) rightSuffixSum[i]++;
                }
                // Total zeros in the array if `val` is 1 (or vice versa)
                int totalSwapsNeeded = rightSuffixSum[0];
                // Track current number of required swaps in the current segment
                int currentSwapCount = 0;
                int minimumSwaps =
                    totalSwapsNeeded - rightSuffixSum[length - totalSwapsNeeded];

                // Iterate to find the minimum swaps by sliding the potential block of grouped `val`
                for (int i = 0; i < totalSwapsNeeded; i++)
                {
                    if (data[i] == (val ^ 1)) currentSwapCount++;
                    int remaining = (totalSwapsNeeded - i - 1);
                    int requiredSwaps =
                        ((i + 1) - currentSwapCount) +
                        (remaining - rightSuffixSum[length - remaining]);
                    minimumSwaps = Math.Min(minimumSwaps, requiredSwaps);
                }
                return minimumSwaps;
            }
            /* Approach 2: Using Sliding Window
            Complexity Analysis
Let n be the length of the array.
•	Time complexity: O(n)
We perform a single pass to count val and then a single pass with the sliding window.
•	Space complexity: O(1)
O(1), since we are using a constant amount of extra space, regardless of the input size.	

             */
            public int UsingSlidingWindow(int[] nums)
            {
                // Calculate the minimum swaps needed to group all 1s or all 0s together
                int op1 = MinSwapsHelper(nums, 0); // Grouping all 0s together
                int op2 = MinSwapsHelper(nums, 1); // Grouping all 1s together
                return Math.Min(op1, op2);
                // Helper function to calculate the minimum swaps required
                // to group all `val` together
                int MinSwapsHelper(int[] data, int val)
                {
                    int length = data.Length;
                    int totalValCount = 0;

                    // Count the total number of `val` in the array
                    for (int i = length - 1; i >= 0; i--)
                    {
                        if (data[i] == val) totalValCount++;
                    }

                    // If there is no `val` or the array is full of `val`,
                    // no swaps are needed
                    if (totalValCount == 0 || totalValCount == length) return 0;

                    int start = 0, end = 0;
                    int maxValInWindow = 0, currentValInWindow = 0;

                    // Initial window setup: count the number of `val` in
                    // the first window of size `totalValCount`
                    while (end < totalValCount)
                    {
                        if (data[end++] == val) currentValInWindow++;
                    }
                    maxValInWindow = Math.Max(maxValInWindow, currentValInWindow);

                    // Slide the window across the array to find the
                    // maximum number of `val` in any window
                    while (end < length)
                    {
                        if (data[start++] == val) currentValInWindow--;
                        if (data[end++] == val) currentValInWindow++;
                        maxValInWindow = Math.Max(maxValInWindow, currentValInWindow);
                    }

                    // Minimum swaps are the total `val` minus
                    // the maximum found in any window
                    return totalValCount - maxValInWindow;
                }
            }
            /* Approach 3: Cleaner and More Intuitive Sliding Window
            Complexity Analysis
Let n be the length of the array.
•	Time complexity: O(n)
The algorithm processes each element of the array once while expanding and sliding the window. Therefore, the time complexity is linear with respect to the number of elements in the array.
•	Space complexity: O(1)
The algorithm uses a constant amount of extra space for variables regardless of the size of the input array. Therefore, the space complexity is constant.

             */
            public int UsingSlidingWindowOptimal(int[] nums)
            {
                // Initialize minimum swaps to a large value
                int minimumSwaps = int.MaxValue;

                // Calculate the total number of 1s in the array
                int totalOnes = 0;
                foreach (int num in nums)
                {
                    totalOnes += num;
                }

                // Initialize the count of 1s in the current window
                int onesCount = nums[0];
                int end = 0;

                // Slide the window across the array
                for (int start = 0; start < nums.Length; ++start)
                {
                    // Adjust onesCount by removing the element that is sliding out of
                    // the window
                    if (start != 0)
                    {
                        onesCount -= nums[start - 1];
                    }

                    // Expand the window to the right until it reaches the desired size
                    while (end - start + 1 < totalOnes)
                    {
                        end++;
                        onesCount += nums[end % nums.Length];
                    }

                    // Update the minimum number of swaps needed
                    minimumSwaps = Math.Min(minimumSwaps, totalOnes - onesCount);
                }

                return minimumSwaps;
            }



        }


        /* 2340. Minimum Adjacent Swaps to Make a Valid Array
        https://leetcode.com/problems/minimum-adjacent-swaps-to-make-a-valid-array/description/
        https://algo.monster/liteproblems/2340.
         */
        class MinimumAdjacentSwapsToMakeAValidArraySol
        {
            /*             Time and Space Complexity
            Time Complexity
            The time complexity of the provided code is primarily determined by the single loop that iterates through the array nums. Since the loop runs for each element in the array, the time complexity is O(n), where n is the length of the array nums.
            Space Complexity
            The space complexity of the code is O(1) because it uses a fixed amount of extra space regardless of the size of the input array. The extra space is used for the variables i, j, k, and v, whose storage does not scale with the size of the input array.
             */
            // Function to find the minimum number of swaps required to make the given array sorted
            public int MinimumSwaps(int[] nums)
            {
                int n = nums.Length; // Length of the given array
                int minIndex = 0, maxIndex = 0; // Initialize indices for minimum and maximum elements

                // Loop through the array to find the indices for the minimum and maximum elements
                for (int k = 0; k < n; ++k)
                {
                    // Update the index of the minimum element found so far
                    if (nums[k] < nums[minIndex] || (nums[k] == nums[minIndex] && k < minIndex))
                    {
                        minIndex = k;
                    }
                    // Update the index of the maximum element found so far
                    if (nums[k] > nums[maxIndex] || (nums[k] == nums[maxIndex] && k > maxIndex))
                    {
                        maxIndex = k;
                    }
                }

                // If the minimum and maximum elements are at the same position, no swaps are needed
                if (minIndex == maxIndex)
                {
                    return 0;
                }

                // Calculate the number of swaps required
                // The calculation is done by considering the positions of the minimum and maximum elements 
                // and adjusting the swap count depending on their relative positions
                int swaps = minIndex + n - 1 - maxIndex - (minIndex > maxIndex ? 1 : 0);

                // Return the number of swaps
                return swaps;
            }
        }

        /* 786. K-th Smallest Prime Fraction
        https://leetcode.com/problems/k-th-smallest-prime-fraction/description/
         */
        class KthSmallestPrimeFractionSol
        {
            /* Approach 1: Binary Search
            Complexity Analysis
Let n be the size of the input array and m be the maximum value in the array.
•	Time complexity: O(n⋅log(m))
The algorithm uses binary search. Within each iteration of the binary search, we perform a linear scan through the array to count the number of fractions smaller than the current mid value. Since the array is sorted, this linear scan takes O(n) time.
Binary search takes O(logx) where x is the number of elements in the search space because each iteration reduces the size of the search space by half. We will stop generating fractions and terminate the search when the total number of smaller fractions equals k. This will happen when the size of the search space becomes smaller than the smallest possible difference between two fractions, which is m21.
This means the size of the search space can be up to m2. Therefore, the total time complexity is O(n⋅log(m2)), which simplifies to O(n⋅log(m)).
•	Space complexity: O(1)
The algorithm uses constant space becuase we only use a constant amount of extra space for storing variables regardless of the input size. We don't use any additional data structures whose size depends on the input size.

             */
            public int[] UsingBinarySearch(int[] arr, int k)
            {
                int n = arr.Length;
                double left = 0, right = 1.0;

                // Binary search for finding the kth smallest prime fraction
                while (left < right)
                {
                    // Calculate the middle value
                    double mid = (left + right) / 2;

                    // Initialize variables to keep track of maximum fraction and indices
                    double maxFraction = 0.0;
                    int totalSmallerFractions = 0, numeratorIdx = 0, denominatorIdx = 0;
                    int j = 1;

                    // Iterate through the array to find fractions smaller than mid
                    for (int i = 0; i < n - 1; i++)
                    {
                        while (j < n && arr[i] >= mid * arr[j])
                        {
                            j++;
                        }

                        // Count smaller fractions
                        totalSmallerFractions += (n - j);

                        // If we have exhausted the array, break
                        if (j == n) break;

                        // Calculate the fraction
                        double fraction = (double)arr[i] / arr[j];

                        // Update maxFraction and indices if necessary
                        if (fraction > maxFraction)
                        {
                            numeratorIdx = i;
                            denominatorIdx = j;
                            maxFraction = fraction;
                        }
                    }

                    // Check if we have found the kth smallest prime fraction
                    if (totalSmallerFractions == k)
                    {
                        return new int[] { arr[numeratorIdx], arr[denominatorIdx] };
                    }
                    else if (totalSmallerFractions > k)
                    {
                        right = mid; // Adjust the range for binary search
                    }
                    else
                    {
                        left = mid; // Adjust the range for binary search
                    }
                }
                return new int[] { }; // Return empty array if kth smallest prime fraction not found
            }

            /* Approach 2: Priority Queue
            Complexity Analysis
Let n be the size of the input array and k be the integer k.
•	Time complexity: O((n+k)⋅logn)
Pushing the initial fractions into the priority queue takes O(nlogn).
Iteratively removing and replacing fractions takes O(klogn) and retrieving the kth smallest fraction takes O(logn).
Thus the overall time complexity of the algorithm is O(nlogn+klogn), which can write as O((n+k)⋅logn)
•	Space complexity: O(n)
The space required by the priority queue to store fractions is O(n) since it can potentially hold all fractions formed by dividing each element by the largest element.
The additional space used by other variables like cur, numeratorIndex, denominatorIndex, etc., is constant and doesn't depend on the size of the input array.
Thus the overall space complexity of the algorithm is O(n).

             */
            public int[] UsingMaxHeap(int[] arr, int k)
            {
                // Create a priority queue to store pairs of fractions
                PriorityQueue<double[], double[]> maxHeap = new PriorityQueue<double[], double[]>(
                    Comparer<double[]>.Create((a, b) => b[0].CompareTo(a[0])));

                // Push the fractions formed by dividing each element by
                // the largest element into the priority queue
                for (int i = 0; i < arr.Length - 1; i++)
                {
                    maxHeap.Enqueue(new double[] {
                -1.0 * arr[i] / arr[arr.Length - 1],
                i,
                arr.Length - 1
            }, new double[] {
                -1.0 * arr[i] / arr[arr.Length - 1],
                i,
                arr.Length - 1
            });
                }

                // Iteratively remove the top element (smallest fraction) 
                // and replace it with the next smallest fraction
                while (--k > 0)
                {
                    double[] cur = maxHeap.Dequeue();
                    int numeratorIndex = (int)cur[1];
                    int denominatorIndex = (int)cur[2] - 1;
                    if (denominatorIndex > numeratorIndex)
                    {
                        maxHeap.Enqueue(new double[] {
                        -1.0 * arr[numeratorIndex] / arr[denominatorIndex],
                        numeratorIndex,
                        denominatorIndex
                }, new double[] {
                        -1.0 * arr[numeratorIndex] / arr[denominatorIndex],
                        numeratorIndex,
                        denominatorIndex
                });
                    }
                }

                // Retrieve the kth smallest fraction from 
                // the top of the priority queue
                double[] result = maxHeap.Dequeue();
                return new int[] { arr[(int)result[1]], arr[(int)result[2]] };
            }
        }

        /* 912. Sort an Array
        https://leetcode.com/problems/sort-an-array/description/
         */
        class SortArraySol
        {/* 
            Approach 1: Merge Sort
Complexity Analysis
Here, n is the number of elements in the nums array.
•	Time complexity: O(nlogn)
o	We divide the nums array into two halves till there is only one element in the array, which will lead to O(logn) steps.
n→n/2→n/4→...→1 (k steps)
n/2(k−1)=1⟹ k≈logn
o	And after each division, we merge those respective halves which will take O(n) time each.
o	Thus, overall it takes O(nlogn) time.
•	Space complexity: O(n)
o	The recursive stack will take O(logn) space and we used an additional array temporaryArray of size n.
o	Thus, overall we use O(logn+n)=O(n) space.
             */
            public int[] UsingMergeSort(int[] nums)
            {
                int[] temporaryArray = new int[nums.Length];
                MergeSort(nums, 0, nums.Length - 1, temporaryArray);
                return nums;
            }
            // Function to merge two sub-arrays in sorted order.
            private void Merge(int[] arr, int left, int mid, int right, int[] tempArr)
            {
                // Calculate the start and sizes of two halves.
                int start1 = left;
                int start2 = mid + 1;
                int n1 = mid - left + 1;
                int n2 = right - mid;

                // Copy elements of both halves into a temporary array.
                for (int idx = 0; idx < n1; idx++)
                {
                    tempArr[start1 + idx] = arr[start1 + idx];
                }
                for (int idx = 0; idx < n2; idx++)
                {
                    tempArr[start2 + idx] = arr[start2 + idx];
                }

                // Merge the sub-arrays 'in tempArray' back into the original array 'arr' in sorted order.
                int i = 0, j = 0, k = left;
                while (i < n1 && j < n2)
                {
                    if (tempArr[start1 + i] <= tempArr[start2 + j])
                    {
                        arr[k] = tempArr[start1 + i];
                        i += 1;
                    }
                    else
                    {
                        arr[k] = tempArr[start2 + j];
                        j += 1;
                    }
                    k += 1;
                }

                // Copy remaining elements
                while (i < n1)
                {
                    arr[k] = tempArr[start1 + i];
                    i += 1;
                    k += 1;
                }
                while (j < n2)
                {
                    arr[k] = tempArr[start2 + j];
                    j += 1;
                    k += 1;
                }
            }

            // Recursive function to sort an array using merge sort
            private void MergeSort(int[] arr, int left, int right, int[] tempArr)
            {
                if (left >= right)
                {
                    return;
                }
                int mid = (left + right) / 2;
                // Sort first and second halves recursively.
                MergeSort(arr, left, mid, tempArr);
                MergeSort(arr, mid + 1, right, tempArr);
                // Merge the sorted halves.
                Merge(arr, left, mid, right, tempArr);
            }

            /* Approach 2: Heap Sort
            Complexity Analysis
Here, n is the number of elements in the nums array.
•	Time complexity: O(nlogn)
o	While heapifying the nums array we traverse the height of the complete binary tree made using n elements, which leads to O(logn) time operations, and the heapify is done n times, once for each element.
o	Thus, overall it takes O(nlogn) time.
•	Space complexity: O(logn)
o	The recursive stack will take O(logn) space, the sorting happens in place and we don't use any other additional memory.

             */
            public int[] UsingHeapSort(int[] nums)
            {
                HeapSort(nums);
                return nums;
            }
            private void Swap(int[] arr, int index1, int index2)
            {
                int temp = arr[index1];
                arr[index1] = arr[index2];
                arr[index2] = temp;
            }

            // Function to heapify a subtree (in top-down order) rooted at index i.
            private void Heapify(int[] arr, int n, int i)
            {
                // Initialize largest as root 'i'.
                int largest = i;
                int left = 2 * i + 1;
                int right = 2 * i + 2;

                // If left child is larger than root.
                if (left < n && arr[left] > arr[largest])
                {
                    largest = left;
                }

                // If right child is larger than largest so far.
                if (right < n && arr[right] > arr[largest])
                {
                    largest = right;
                }

                // If largest is not root swap root with largest element
                // Recursively heapify the affected sub-tree (i.e. move down).
                if (largest != i)
                {
                    Swap(arr, i, largest);
                    Heapify(arr, n, largest);
                }
            }

            private void HeapSort(int[] arr)
            {
                int n = arr.Length;
                // Build heap; heapify (top-down) all elements except leaf nodes.
                for (int i = n / 2 - 1; i >= 0; i--)
                {
                    Heapify(arr, n, i);
                }

                // Traverse elements one by one, to move current root to end, and
                for (int i = n - 1; i >= 0; i--)
                {
                    Swap(arr, 0, i);
                    // call max heapify on the reduced heap.
                    Heapify(arr, i, 0);
                }
            }
            /* Approach 3: Counting Sort
            Complexity Analysis
            Here, n is the number of elements in the nums array, and k is the range of value of its elements (minimum value to maximum value).
            •	Time complexity: O(n+k)
            o	We iterate on the array elements while counting the frequency and finding minimum and maximum values, thus taking O(n) time.
            o	Then we iterate on the input array's element's range which will take O(k) time.
            o	Thus, overall it takes O(n+k) time.
            •	Space complexity: O(n)
            o	We use a hash map counts which might store all O(n) elements of the input array in worst-case.

             */
            public int[] UsingCountingSort(int[] nums)
            {
                CountingSort(nums);
                return nums;
            }
            private void CountingSort(int[] arr)
            {
                // Create the counting hash map.
                Dictionary<int, int> counts = new();
                int minVal = arr[0], maxVal = arr[0];

                // Find the minimum and maximum values in the array,
                // and update it's count in the hash map.
                for (int i = 0; i < arr.Length; i++)
                {
                    minVal = Math.Min(minVal, arr[i]);
                    maxVal = Math.Max(maxVal, arr[i]);
                    counts[arr[i]] = counts.GetValueOrDefault(arr[i], 0) + 1;
                }

                int index = 0;
                // Place each element in its correct position in the array.
                for (int val = minVal; val <= maxVal; ++val)
                {
                    // Append all 'val's together if they exist.
                    while (counts.GetValueOrDefault(val, 0) > 0)
                    {
                        arr[index] = val;
                        index += 1;
                        counts[val] = counts[val] - 1;
                    }
                }
            }

            /* Approach 4: Radix Sort
Complexity Analysis
Here, n is the number of elements in the nums array, d is the maximum number of digits and b is the size of the bucket used.
•	Time complexity: O(d⋅(n+b))
o	We iterate on the array elements to find the maximum number and then find the count of its digits, thus taking O(n+d) time.
o	Then we sort the array for each integer place which will take O(n+b) time, thus for all d places it will take O(d⋅(n+b)) time.
o	At the end, we seperate out positive and negative numbers and reverse the negatives, which overall will take O(n) time.
o	Thus, overall it takes O((n+d)+d⋅(n+b)+n)=O(d⋅(n+b)) time.
•	Space complexity: O(n+b)
o	We use additional arrays negatives and positives which use O(n) space and buckets which use O(n+b) space.

             */
            // Bucket sort function for each place value digit.
            public int[] UsingRadixSort(int[] numbers)
            {
                RadixSort(numbers);
                return numbers;
            }
            private void BucketSort(int[] array, int placeValue)
            {
                List<List<int>> buckets = new List<List<int>>(10);
                for (int digit = 0; digit < 10; ++digit)
                {
                    buckets.Add(new List<int>());
                }

                // Store the respective number based on its digit.
                foreach (int value in array)
                {
                    int digit = Math.Abs(value) / placeValue;
                    digit = digit % 10;
                    buckets[digit].Add(value);
                }

                // Overwrite 'array' in sorted order of current place digits.
                int index = 0;
                for (int digit = 0; digit < 10; ++digit)
                {
                    foreach (int value in buckets[digit])
                    {
                        array[index] = value;
                        index++;
                    }
                }
            }

            // Radix sort function.
            private void RadixSort(int[] array)
            {
                // Find the absolute maximum element to find max number of digits.
                int maxElement = array[0];
                foreach (int value in array)
                {
                    maxElement = Math.Max(Math.Abs(value), maxElement);
                }
                int maxDigits = 0;
                while (maxElement > 0)
                {
                    maxDigits += 1;
                    maxElement /= 10;
                }

                // Radix sort, least significant digit place to most significant.
                int placeValue = 1;
                for (int digit = 0; digit < maxDigits; ++digit)
                {
                    BucketSort(array, placeValue);
                    placeValue *= 10;
                }

                // Separate out negatives and reverse them. 
                List<int> negatives = new List<int>();
                List<int> positives = new List<int>();
                foreach (int value in array)
                {
                    if (value < 0)
                    {
                        negatives.Add(value);
                    }
                    else
                    {
                        positives.Add(value);
                    }
                }
                negatives.Reverse();

                // Final 'answer' will be 'negative' elements, then 'positive' elements.
                int index = 0;
                foreach (int value in negatives)
                {
                    array[index] = value;
                    index++;
                }
                foreach (int value in positives)
                {
                    array[index] = value;
                    index++;
                }
            }


        }

        /* 1442. Count Triplets That Can Form Two Arrays of Equal XOR
        https://leetcode.com/problems/count-triplets-that-can-form-two-arrays-of-equal-xor/description/
         */

        class CountTripletsThatFormTwoArraysOfEqualXORSol
        {

            /* Approach 1: Brute Force With Prefix
            Complexity Analysis
            Let n be the size of the input array.
            •	Time complexity: O(n^3)
            There are three nested loops, each iterating over the entire array, resulting in a time complexity of O(n⋅n⋅n)=O(n^3).
            •	Space complexity: O(1)
            We only use a few variables (count, xorA, xorB) to store intermediate results, which take constant space.

             */
            public int NaiveWithPrefix(int[] arr)
            {
                int count = 0;

                // Iterate over each possible starting index i
                for (int start = 0; start < arr.Length - 1; ++start)
                {
                    // Initialize XOR value for the subarray from start to mid-1
                    int xorA = 0;

                    // Iterate over each possible middle index j
                    for (int mid = start + 1; mid < arr.Length; ++mid)
                    {
                        // Update xorA to include arr[mid - 1]
                        xorA ^= arr[mid - 1];

                        // Initialize XOR value for the subarray from mid to end
                        int xorB = 0;

                        // Iterate over each possible ending index k (starting from mid)
                        for (int end = mid; end < arr.Length; ++end)
                        {
                            // Update xorB to include arr[end]
                            xorB ^= arr[end];

                            // found a valid triplet (start, mid, end)
                            if (xorA == xorB)
                            {
                                ++count;
                            }
                        }
                    }
                }

                return count;
            }
            /*             Approach 2: Nested Prefix XOR
            Complexity Analysis
            Let n be the size of the input array.
            •	Time complexity: O(n^2)
            There are two nested loops, each iterating over the array, resulting in a time complexity of O(n⋅n)=O(n^2).
            •	Space complexity: O(n)
            We create a new array prefixXOR of the same size as the input array, taking O(n) space.
            Note: This approach could be implemented with O(1) space by modifying the original array.

             */
            public int UsingNestedPrefixXOR(int[] arr)
            {
                int[] prefixXOR = new int[arr.Length + 1];
                prefixXOR[0] = 0;
                Array.Copy(arr, 0, prefixXOR, 1, arr.Length);
                int size = prefixXOR.Length;

                // Perform XOR on consecutive elements in the modified array
                for (int i = 1; i < size; i++)
                {
                    prefixXOR[i] ^= prefixXOR[i - 1];
                }

                int count = 0;

                // Iterate through the modified array to count triplets
                for (int start = 0; start < size; start++)
                {
                    for (int end = start + 1; end < size; end++)
                    {
                        if (prefixXOR[start] == prefixXOR[end])
                        {
                            // Increment the result by the count of valid triplets
                            count += end - start - 1;
                        }
                    }
                }

                return count;
            }
            /* Approach 3: Two Pass Prefix XOR
            Complexity Analysis
Let n be the size of the input array.
•	Time complexity: O(n)
There are two different loops iterating over the array, resulting in a time complexity of O(2⋅n), which can be simplified to O(n).
•	Space complexity: O(n)
In the worst case, each element in the array can have a unique XOR value, requiring O(n) space to store the counts and totals in the maps.	

             */
            public int UsingTwoPassPrefixXOR(int[] arr)
            {
                int[] prefixXOR = new int[arr.Length + 1];
                prefixXOR[0] = 0;
                Array.Copy(arr, 0, prefixXOR, 1, arr.Length);
                int size = prefixXOR.Length;
                int count = 0;

                // Performing XOR operation on the array elements
                for (int i = 1; i < size; ++i) prefixXOR[i] ^= prefixXOR[i - 1];

                // Maps to store counts and totals of XOR values encountered
                Dictionary<int, int> countMap = new();
                Dictionary<int, int> totalMap = new();

                // Iterating through the array
                for (int i = 0; i < size; ++i)
                {
                    // Calculating contribution of current element to the result
                    count +=
                    countMap.GetValueOrDefault(prefixXOR[i], 0) * (i - 1) -
                    totalMap.GetValueOrDefault(prefixXOR[i], 0);

                    // Updating total count of current XOR value
                    totalMap[
                        prefixXOR[i]] =
                        totalMap.GetValueOrDefault(prefixXOR[i], 0) + i
                    ;
                    countMap[prefixXOR[i]] =
                        countMap.GetValueOrDefault(prefixXOR[i], 0) + 1
                    ;
                }

                return count;
            }
            /* Approach 4: One Pass Prefix XOR
Complexity Analysis
Let n be the size of the input array.
•	Time complexity: O(n)
There is only a single loop iterating over the array, resulting in a time complexity of O(n).
•	Space complexity: O(n)
In the worst case, each element in the array can have a unique XOR value, requiring O(n) space to store the counts and totals in the maps.

             */
            public int UsingOnePassPrefixXOR(int[] arr)
            {
                int size = arr.Length;
                int count = 0;
                int prefix = 0;

                // Maps to store counts and totals of XOR values encountered
                Dictionary<int, int> countMap = new();
                countMap[0] = 1;
                Dictionary<int, int> totalMap = new();

                // Iterating through the array
                for (int i = 0; i < size; ++i)
                {
                    // Calculating XOR prefix
                    prefix ^= arr[i];

                    // Calculating contribution of current element to the result
                    count +=
                    countMap.GetValueOrDefault(prefix, 0) * i -
                    totalMap.GetValueOrDefault(prefix, 0);

                    // Updating total count of current XOR value
                    totalMap[prefix] = totalMap.GetValueOrDefault(prefix, 0) + i + 1;
                    countMap[prefix] = countMap.GetValueOrDefault(prefix, 0) + 1;
                }

                return count;
            }

        }


        /* 1838. Frequency of the Most Frequent Element
        https://leetcode.com/problems/frequency-of-the-most-frequent-element/description/
         */
        class FrequencyOfMostFrequentElementSol
        {

            /* Approach 1: Sliding Window
            Complexity Analysis
            Given n as the length of nums,
            •	Time complexity: O(n⋅logn)
            Despite the while loop, each iteration of the for loop is amortized O(1). The while loop only runs O(n) times across all iterations. This is because each iteration of the while loop increments left. As left can only increase and cannot exceed n, the while loop never performs more than n iterations total. This means the sliding window process runs in O(n).
            However, we need to sort the array, which costs O(n⋅logn).
            •	Space Complexity: O(logn) or O(n)
            We only use a few integer variables, but some space is used to sort.
            The space complexity of the sorting algorithm depends on the implementation of each programming language:
            o	In Java, Arrays.sort() for primitives is implemented using a variant of the Quick Sort algorithm, which has a space complexity of O(logn)
            o	In C++, the sort() function provided by STL uses a hybrid of Quick Sort, Heap Sort and Insertion Sort, with a worst case space complexity of O(logn)
            o	In Python, the sort() function is implemented using the Timsort algorithm, which has a worst-case space complexity of O(n)

             */
            public int UsingSlidingWindow(int[] nums, int k)
            {
                Array.Sort(nums);
                int left = 0;
                int ans = 0;
                long curr = 0;

                for (int right = 0; right < nums.Length; right++)
                {
                    long target = nums[right];
                    curr += target;

                    while ((right - left + 1) * target - curr > k)
                    {
                        curr -= nums[left];
                        left++;
                    }

                    ans = Math.Max(ans, right - left + 1);
                }

                return ans;
            }

            /* Approach 2: Advanced Sliding Window
            Complexity Analysis
            Given n as the length of nums,
            •	Time complexity: O(n⋅logn)
            Each iteration of the for loop costs O(1). This means the sliding window process runs in O(n).
            However, we need to sort the array, which costs O(n⋅logn).
            •	Space Complexity: O(logn) or O(n)
            We only use a few integer variables, but some space is used to sort.
            The space complexity of the sorting algorithm depends on the implementation of each programming language:
            o	In Java, Arrays.sort() for primitives is implemented using a variant of the Quick Sort algorithm, which has a space complexity of O(logn)
            o	In C++, the sort() function provided by STL uses a hybrid of Quick Sort, Heap Sort and Insertion Sort, with a worst case space complexity of O(logn)
            o	In Python, the sort() function is implemented using the Timsort algorithm, which has a worst-case space complexity of O(n)

             */
            public int UsingSlidingWindowAdavanced(int[] nums, int k)
            {
                Array.Sort(nums);
                int left = 0;
                long curr = 0;

                for (int right = 0; right < nums.Length; right++)
                {
                    long target = nums[right];
                    curr += target;

                    if ((right - left + 1) * target - curr > k)
                    {
                        curr -= nums[left];
                        left++;
                    }
                }

                return nums.Length - left;
            }
            /* Approach 3: Binary Search 
Complexity Analysis
Given n as the length of nums,
•	Time complexity: O(n⋅logn)
First, we sort nums which costs O(n⋅logn).
Next, we iterate over the indices of nums. For each of the O(n) indices, we call check, which costs up to O(logn) as its a binary search over the array's elements. The total cost is O(n⋅logn).
•	Space complexity: O(n)
The prefix array uses O(n) space.	

            */
            public int UsingBinarySearch(int[] nums, int k)
            {
                Array.Sort(nums);
                long[] prefix = new long[nums.Length];
                prefix[0] = nums[0];

                for (int i = 1; i < nums.Length; i++)
                {
                    prefix[i] = nums[i] + prefix[i - 1];
                }

                int ans = 0;
                for (int i = 0; i < nums.Length; i++)
                {
                    ans = Math.Max(ans, Check(i, k, nums, prefix));
                }

                return ans;

            }
            public int Check(int i, int k, int[] nums, long[] prefix)
            {
                int target = nums[i];
                int left = 0;
                int right = i;
                int best = i;

                while (left <= right)
                {
                    int mid = (left + right) / 2;
                    long count = i - mid + 1;
                    long finalSum = count * target;
                    long originalSum = prefix[i] - prefix[mid] + nums[mid];
                    long operationsRequired = finalSum - originalSum;

                    if (operationsRequired > k)
                    {
                        left = mid + 1;
                    }
                    else
                    {
                        best = mid;
                        right = mid - 1;
                    }
                }

                return i - best + 1;
            }



        }

        /* 2191. Sort the Jumbled Numbers
        https://leetcode.com/problems/sort-the-jumbled-numbers/description/
         */
        class SortJumbledSol
        {
            /* Approach 1: Conversion using strings and Sorting 
Complexity Analysis
Let n be the size of the nums array.
•	Time complexity: O(n⋅logn)
For every integer in nums, we convert it to a string and perform constant operations over its length. The time taken for converting an integer to a string, and vice versa, is O(lengthofinteger) time, which is proportional to the logarithmic value of n. Therefore, the time complexity for these operations is given by O(n⋅logn).
Sorting the array of pairs takes O(n⋅logn) time. All other operations are linear or constant time.
Therefore, the total time complexity is given by O(n⋅logn).
•	Space complexity: O(n)
We create two new arrays of size n. Apart from this, some extra space is used when we sort arrays in place. The space complexity of the sorting algorithm depends on the programming language.
o	In Python, the sort method sorts a list using the Timsort algorithm which is a combination of Merge Sort and Insertion Sort and has O(n) additional space.
o	In Java, Arrays.sort() is implemented using a variant of the Quick Sort algorithm which has a space complexity of O(logn) for sorting two arrays.
o	In C++, the sort() function is implemented as a hybrid of Quick Sort, Heap Sort, and Insertion Sort, with a worse-case space complexity of O(logn).
Therefore, the total space complexity is given by O(n).

            */
            public int[] UsingConversionWithStringsAndSorting(int[] mapping, int[] nums)
            {
                List<int[]> storePairs = new List<int[]>();

                for (int i = 0; i < nums.Length; ++i)
                {
                    // Convert current value to string
                    string number = nums[i].ToString();
                    string formed = "";
                    for (int j = 0; j < number.Length; ++j)
                    {
                        formed += mapping[number[j] - '0'].ToString();
                    }
                    // Store the mapped value.
                    int mappedValue = int.Parse(formed);
                    // Push a pair consisting of mapped value and original value's index.
                    storePairs.Add(new int[] { mappedValue, i });
                }

                // Sort the array in non-decreasing order by the first value (default).
                storePairs.Sort((o1, o2) => o1[0].CompareTo(o2[0]));

                int[] answer = new int[nums.Length];
                for (int i = 0; i < storePairs.Count; i++)
                {
                    answer[i] = nums[storePairs[i][1]];
                }
                return answer;
            }

            /* Approach 2: Conversion without using strings and Sorting
Complexity Analysis
Let n be the size of the nums array.
•	Time complexity: O(n⋅logn)
For every integer in nums, we convert it to the mapped integer. The time taken for this operation is O(lengthofinteger) time, which is proportional to the logarithmic value of n. Therefore, the time complexity for these operations on nums is given by O(n⋅logn).
Sorting the array of pairs takes O(n⋅logn) time. All other operations are linear or constant time.
Therefore, the total time complexity is given by O(n⋅logn).
•	Space complexity: O(n)
We create two new arrays of size n. Apart from this, some extra space is used when we sort arrays in place. The space complexity of the sorting algorithm depends on the programming language.
o	In Python, the sort method sorts a list using the Timsort algorithm which is a combination of Merge Sort and Insertion Sort and has O(n) additional space.
o	In Java, Arrays.sort() is implemented using a variant of the Quick Sort algorithm which has a space complexity of O(logn) for sorting two arrays.
o	In C++, the sort() function is implemented as a hybrid of Quick Sort, Heap Sort, and Insertion Sort, with a worse-case space complexity of O(logn).
Therefore, the total space complexity is given by O(n).

             */
            public int[] UsingConversionWithoutStringsAndSorting(int[] mapping, int[] nums)
            {
                List<int[]> storePairs = new();

                for (int i = 0; i < nums.Length; i++)
                {
                    int mappedValue = 0;
                    int temp = nums[i];
                    int place = 1;

                    if (temp == 0)
                    {
                        storePairs.Add(new int[] { mapping[0], i });
                        continue;
                    }
                    while (temp != 0)
                    {
                        mappedValue = place * mapping[temp % 10] + mappedValue;
                        place *= 10;
                        temp /= 10;
                    }
                    storePairs.Add(new int[] { mappedValue, i });
                }

                storePairs.Sort((a, b) => a[0] - b[0]);

                int[] answer = new int[nums.Length];
                for (int i = 0; i < storePairs.Count; i++)
                {
                    answer[i] = nums[storePairs[i][1]];
                }

                return answer;
            }

        }

        /* 
        2602. Minimum Operations to Make All Array Elements Equal
        https://leetcode.com/problems/minimum-operations-to-make-all-array-elements-equal/description/	
        https://algo.monster/liteproblems/2602
         */
        class MinOperationsToMakeAllArrayElementsEqualSol
        {
            /* 
Time and Space Complexity
The given algorithm involves sorting the array nums and performing binary searches for each query in queries.
Time Complexity
1.	Sorting the nums array has a time complexity of O(n \log n) where n is the length of the nums array.
2.	The accumulate function is linear, contributing O(n) to the time complexity.
3.	For each query, a binary search is performed twice using bisect_left, which has a time complexity of O(\log n). Since there are q queries, the total complexity for this part is O(q \log n).
The total time complexity combines these contributions, resulting in O(n \log n) + O(n) + O(q \log n). Since O(n \log n) dominates O(n), and the number of queries q could vary independently of n, the overall time complexity is O((n + q) \log n).
However, since the reference answer only specifies O(n \log n), it implies that n is the dominant term, and q is expected to be not significantly larger than n.
Space Complexity
1.	The sorted array is a modification in place, so it does not add to the space complexity.
2.	The s variable is a list storing the cumulative sum, contributing O(n) to the space complexity.
3.	A constant amount of extra space is used for variables i, t, and the iterative variables, which does not depend on the size of the input.
Thus, the overall space complexity is O(n).	

             */
            public List<long> MinOperations(int[] nums, int[] queries)
            {
                // Sort the array of numbers in non-decreasing order
                Array.Sort(nums);

                int numberOfElements = nums.Length; // Number of elements in nums
                                                    // Create a prefix sum array with an extra space for ease of calculations
                long[] prefixSum = new long[numberOfElements + 1];

                // Calculate the prefix sum
                for (int i = 0; i < numberOfElements; ++i)
                {
                    prefixSum[i + 1] = prefixSum[i] + nums[i];
                }

                // Initialize the answer list
                List<long> answerList = new List<long>();

                // Process each query
                foreach (int query in queries)
                {
                    // Search for the first element in nums that is greater than or equal to query + 1
                    int index = BinarySearch(nums, query + 1);
                    // Calculate the total sum needed to reduce elements larger or equal to query + 1 to query
                    long totalOperations = prefixSum[numberOfElements] - prefixSum[index] - 1L * (numberOfElements - index) * query;

                    // Search for the first element in nums that is greater than or equal to query
                    index = BinarySearch(nums, query);
                    // Add the total sum needed to increase elements smaller than query to query
                    totalOperations += 1L * query * index - prefixSum[index];

                    // Add the calculated operations for the current query to the answer list
                    answerList.Add(totalOperations);
                }

                return answerList; // Return the answer list
            }

            // Helper method to perform binary search
            private int BinarySearch(int[] nums, int x)
            {
                int left = 0;
                int right = nums.Length;
                // Binary search to find the index of the first number greater or equal to x
                while (left < right)
                {
                    int mid = (left + right) >> 1; // Middle position
                    if (nums[mid] >= x)
                    {
                        right = mid;
                    }
                    else
                    {
                        left = mid + 1;
                    }
                }
                return left; // Return the first index where nums[index] >= x
            }
        }

        /* 624. Maximum Distance in Arrays
        https://leetcode.com/problems/maximum-distance-in-arrays/description/
         */
        class MaxDistanceInArraysSol
        {
            /* 
            Approach #1 Brute Force [Time Limit Exceeded]
Complexity Analysis
•	Time complexity: O((n∗x)^2). We traverse over all the arrays in arrays for every element of every array considered. Here, n refers to the number of arrays in arrays and x refers to the average number of elements in each array in arrays.
•	Space complexity: O(1). Constant extra space is used.

             */
            public int Naive(List<List<int>> arrays)
            {
                int res = 0;
                int n = arrays.Count;
                for (int i = 0; i < n - 1; i++)
                {
                    for (int j = 0; j < arrays[i].Count; j++)
                    {
                        for (int k = i + 1; k < n; k++)
                        {
                            for (int l = 0; l < arrays[k].Count; l++)
                            {
                                res = Math.Max(res, Math.Abs(arrays[i][j] - arrays[k][l]));
                            }
                        }
                    }
                }
                return res;
            }
            /* Approach #2 Better Brute Force [Time Limit Exceeded]
Complexity Analysis
•	Time complexity: O(n^2). We consider only max and min values directly for every array currenty considered. Here, n refers to the number of arrays in arrays.
•	Space complexity: O(1). Constant extra space is used.

             */
            public int NaiveOptimal(List<List<int>> arrays)
            {
                List<int> array1, array2;
                int res = 0;
                int n = arrays.Count;
                for (int i = 0; i < n - 1; i++)
                {
                    for (int j = i + 1; j < n; j++)
                    {
                        array1 = arrays[i];
                        array2 = arrays[j];
                        res = Math.Max(res, Math.Abs(array1[0] - array2[array2.Count - 1]));
                        res = Math.Max(res, Math.Abs(array2[0] - array1[array1.Capacity - 1]));
                    }
                }
                return res;
            }
            /* Approach #3 Single Scan [Accepted]
Complexity Analysis
•	Time complexity: O(n). We traverse over arrays of length n once only.
•	Space complexity: O(1). Constant extra space is used.
            */
            public int UsingSingleScan(List<List<int>> arrays)
            {
                int res = 0;
                int n = arrays[0].Count;
                int min_val = arrays[0][0];
                int max_val = arrays[0][arrays[0].Count - 1];
                for (int i = 1; i < arrays.Count; i++)
                {
                    n = arrays[i].Count;
                    res = Math.Max(res, Math.Max(Math.Abs(arrays[i][n - 1] - min_val),
                                                 Math.Abs(max_val - arrays[i][0])));
                    min_val = Math.Min(min_val, arrays[i][0]);
                    max_val = Math.Max(max_val, arrays[i][n - 1]);
                }
                return res;
            }
        }


        /* 852. Peak Index in a Mountain Array
        https://leetcode.com/problems/peak-index-in-a-mountain-array/description/
         */

        class PeakIndexInMountainArraySol
        {
            /* Approach 1: Linear Scan
Complexity Analysis
Here n is the length of arr.
•	Time complexity: O(n).
o	We are doing a linear scan and comparing adjacent elements until we get the peak of the mountain over the arr array. In the worst-case situation, the peak of the mountain could correspond to the second last element of arr, in which case we would take O(n) time.
•	Space complexity: O(1).
o	We are not using any extra space other than an integer i, which takes up constant space.

             */
            public int UsingLinearScan(int[] arr)
            {
                int i = 0;
                while (arr[i] < arr[i + 1])
                {
                    i++;
                }
                return i;
            }
            /* Approach 2: Binary Search
Complexity Analysis
Here n is the length of arr.
•	Time complexity: O(logn).
o	We perform O(logn) iterations using the binary search algorithm as the problem set is divided into half in each iteration.
•	Space complexity: O(1).
o	Except for a few variables l, r, and mid which take constant space each, we do not consume any other space.

             */
            public int UsingBinarySearch(int[] arr)
            {
                int l = 0, r = arr.Length - 1, mid;
                while (l < r)
                {
                    mid = (l + r) / 2;
                    if (arr[mid] < arr[mid + 1])
                        l = mid + 1;
                    else
                        r = mid;
                }
                return l;
            }
        }

        /* 932. Beautiful Array	
        https://leetcode.com/problems/beautiful-array/description/
         */

        class BeautifulArraySol
        {

            /*
            Approach 1: Divide and Conquer
                     Complexity Analysis
            •	Time Complexity: O(NlogN). The function f is called only O(logN) times, and each time does O(N) work.
            •	Space Complexity: O(NlogN).
             */
            Dictionary<int, int[]> memo;
            public int[] UsingDivideAndConquer(int N)
            {
                memo = new();
                return f(N);
            }

            public int[] f(int N)
            {
                if (memo.ContainsKey(N))
                    return memo[N];

                int[] ans = new int[N];
                if (N == 1)
                {
                    ans[0] = 1;
                }
                else
                {
                    int t = 0;
                    foreach (int x in f((N + 1) / 2))  // odds
                        ans[t++] = 2 * x - 1;
                    foreach (int x in f(N / 2))  // evens
                        ans[t++] = 2 * x;
                }
                memo[N] = ans;
                return ans;
            }
        }

        /* 137. Single Number II
        https://leetcode.com/problems/single-number-ii/description/
         */
        public class SingleNumberIISol
        {
            /* Approach 1: Sorting
Complexity Analysis
Let N be the length of nums
•	Time complexity: O(NlogN)
Sorting can be optimally done in O(NlogN) time.
After sorting, we are traversing the array once. It may take O(N) time.
Thus, the overall time complexity is O(N+NlogN) which is O(NlogN).
•	Space complexity: O(N).
It depends on the sorting algorithm. Depending on the programming language, sorting algorithms may need O(N) or O(logN) space.

             */
            public int UsingSorting(int[] nums)
            {
                Array.Sort(nums);
                for (int i = 0; i < nums.Length - 1; i += 3)
                {
                    if (nums[i] == nums[i + 1])
                    {
                        continue;
                    }
                    else
                    {
                        return nums[i];
                    }
                }

                return nums[nums.Length - 1];
            }


            /* Approach 2: Hash Map
Complexity Analysis
Let N be the length of nums
•	Time complexity: O(N)
Building counter from nums can be done in O(N) time.
Iterating over the counter can be done in O(N) time.
Thus, overall time complexity is O(N+N) which is O(N).
•	Space complexity: O(N).
We are using a counter to store the frequency of integers. There will be approximately 3N such integers. So, the space complexity is O(3N) which is O(N).

             */
            public int UsingDict(int[] nums)
            {
                Dictionary<int, int> freq = new Dictionary<int, int>();
                foreach (int num in nums)
                {
                    if (!freq.ContainsKey(num))
                    {
                        freq[num] = 1;
                    }
                    else
                    {
                        freq[num]++;
                    }
                }

                int loner = 0;
                foreach (KeyValuePair<int, int> entry in freq)
                {
                    if (entry.Value == 1)
                    {
                        loner = entry.Key;
                        break;
                    }
                }

                return loner;
            }
            /* Approach 3: Mathematics
Complexity Analysis
Let N be the length of nums
•	Time complexity: O(N)
Building a set from nums can be done in O(N) time.
Iterating over the set can be done in O(N) time.
Thus, overall time complexity is O(N+N) which is O(N).
•	Space complexity: O(N).
We are using a set to store the integers. There will be approximately 3N such integers. So, the space complexity is O(3N) which is O(N).

             */
            public int UsingMaths(int[] nums)
            {
                long sumNums = 0;
                HashSet<long> numsSet = new HashSet<long>();
                foreach (int num in nums)
                {
                    numsSet.Add((long)num);
                    sumNums += num;
                }

                long sumSet = 0;
                foreach (long num in numsSet)
                {
                    sumSet += num;
                }

                return (int)((3 * sumSet - sumNums) / 2);
            }

            /* Approach 4: Bit Manipulation
            Complexity Analysis
Let N be the length of nums
•	Time complexity: O(N)
We iterate over all integers in nums once, and for each integer, we iterate over all 32 bits. So, the time complexity is O(32N), which is O(N).
•	Space complexity: O(1)
We use constant extra space, with fixed-size variables. So, the space complexity is O(1).
Note: Reader may comment that time complexity is O(NlogN) because we are iterating the number of times equal to the number of bits in an integer. However, our code is designed in such a way that it will iterate only 32 times, which is constant. So, the time complexity is O(N). The logN factor will not be counted because of our design. We were able to design it because of some pre-knowledge of the problem.

             */
            public int UsingBitManip(int[] nums)
            {
                // Loner
                int loner = 0;
                // Iterate over all bits
                for (int shift = 0; shift < 32; shift++)
                {
                    int bitSum = 0;
                    // For this bit, iterate over all integers
                    foreach (int num in nums)
                    {
                        // Compute the bit of num, and add it to bitSum
                        bitSum += (num >> shift) & 1;
                    }

                    // Compute the bit of loner and place it
                    int lonerBit = bitSum % 3;
                    loner = loner | (lonerBit << shift);
                }

                return loner;
            }
            /* Approach 5: Equation for Bitmask
            Complexity Analysis
Let N be the length of nums
•	Time complexity: O(N)
o	We iterate through nums once.
o	For each integer, we update seenOnce and seenTwice using derived equations. This takes constant time.
Thus, for one num, we take constant time. For N nums, we take O(N) time.
•	Space complexity: O(1)
We use constant extra space for seenOnce and seenTwice.

             */
            public int UsingBitMasking(int[] nums)
            {
                // Initialize seenOnce and seenTwice to 0
                int seenOnce = 0, seenTwice = 0;

                // Iterate through nums
                foreach (int num in nums)
                {
                    // Update using derived equations
                    seenOnce = (seenOnce ^ num) & (~seenTwice);
                    seenTwice = (seenTwice ^ num) & (~seenOnce);
                }

                // Return integer which appears exactly once
                return seenOnce;
            }
            /*             Approach 6: Boolean Algebra and Karnaugh Map
Complexity Analysis
Let N be the length of nums
•	Time complexity: O(N)
We iterate over all num in nums once. During each iteration, we do constant time operations of updating msb and lsb. So, the total time complexity is O(N)
•	Space complexity: O(1)
We use constant space for msb and lsb to store count. So, total space complexity is O(1)

             */
            public int UsingBooleanAlgegraAndKarnaughMap(int[] nums)
            {
                // Count (modulo 3) bits
                int msb = 0, lsb = 0;
                // Process Every Num and update count bits
                foreach (int num in nums)
                {
                    int new_lsb = (~msb & ~lsb & num) | (lsb & ~num);
                    int new_msb = (lsb & num) | (msb & ~num);
                    lsb = new_lsb;
                    msb = new_msb;
                }

                // Return lsb as the answer
                return lsb;
            }


        }

        /* 260. Single Number III
        https://leetcode.com/problems/single-number-iii/description/
         */
        class SingleNumberIIISol
        {
            /*             Approach 1: Hashmap
            Complexity Analysis
            •	Time complexity : O(N) to iterate over the input array.
            •	Space complexity : O(N) to keep the hashmap of N elements.

             */
            public int[] UsingDict(int[] nums)
            {
                Dictionary<int, int> hashmap = new();
                foreach (int n in nums)
                    hashmap[n] = hashmap.GetValueOrDefault(n, 0) + 1;

                int[] output = new int[2];
                int idx = 0;
                foreach (var item in hashmap)
                    if (item.Value == 1) output[idx++] = item.Key;

                return output;
            }
            /* Approach 2: Two bitmasks 
            Complexity Analysis
•	Time complexity : O(N) to iterate over the input array.
•	Space complexity : O(1), it's a constant space solution.

            */
            public int[] UsingTwoBitMasks(int[] nums)
            {
                // difference between two numbers (x and y) which were seen only once
                int bitmask = 0;
                foreach (int num in nums) bitmask ^= num;

                // rightmost 1-bit diff between x and y
                int diff = bitmask & (-bitmask);

                int x = 0;
                // bitmask which will contain only x
                foreach (int num in nums) if ((num & diff) != 0) x ^= num;

                return new int[] { x, bitmask ^ x };
            }

        }


        /* 416. Partition Equal Subset Sum
        https://leetcode.com/problems/partition-equal-subset-sum/description/
         */
        class CanPartitionSol
        {
            /*             
            Approach 1: Brute Force	
            Complexity Analysis
            •	Time Complexity : O(2^n), where n is equal to number of array elements.
            The recursive solution takes the form of a binary tree where there are 2 possibilities for every array element and the maximum depth of the tree could be n. The time complexity is exponential, hence this approach is exhaustive and results in Time Limit Exceeded (TLE).
            •	Space Complexity: O(N) This space will be used to store the recursion stack. We can’t have more than n recursive calls on the call stack at any time.
             */
            public bool Naive(int[] nums)
            {
                int totalSum = 0;
                // find sum of all array elements
                foreach (int num in nums)
                {
                    totalSum += num;
                }
                // if totalSum is odd,it cannot be partitioned into equal sum subset
                if (totalSum % 2 != 0) return false;
                int subSetSum = totalSum / 2;
                int n = nums.Length;
                return Dfs(nums, n - 1, subSetSum);
            }

            private bool Dfs(int[] nums, int n, int subSetSum)
            {
                // Base Cases
                if (subSetSum == 0)
                    return true;
                if (n == 0 || subSetSum < 0)
                    return false;
                return Dfs(nums, n - 1, subSetSum - nums[n - 1]) || Dfs(nums, n - 1, subSetSum);
            }
            /* Approach 2: Top Down Dynamic Programming - Memoization
Complexity Analysis
Let n be the number of array elements and m be the subSetSum.
•	Time Complexity : O(m⋅n).
o	In the worst case where there is no overlapping calculation, the maximum number of entries in the memo would be m⋅n. For each entry, overall we could consider that it takes constant time, i.e. each invocation of dfs() at most emits one entry in the memo.
o	The overall computation is proportional to the number of entries in memo. Hence, the overall time complexity is O(m⋅n).
•	Space Complexity: O(m⋅n). We are using a 2 dimensional array memo of size (m⋅n) and O(n) space to store the recursive call stack. This gives us the space complexity as O(n) + O(m⋅n) = O(m⋅n)

             */
            public bool TopDownDPWithMemo(int[] nums)
            {
                int totalSum = 0;
                // find sum of all array elements
                foreach (int num in nums)
                {
                    totalSum += num;
                }
                // if totalSum is odd, it cannot be partitioned into equal sum subset
                if (totalSum % 2 != 0) return false;
                int subSetSum = totalSum / 2;
                int n = nums.Length;
                Boolean[][] memo = new Boolean[n + 1][];
                return Dfs(nums, n - 1, subSetSum, memo);
            }

            public bool Dfs(int[] nums, int n, int subSetSum, Boolean[][] memo)
            {
                // Base Cases
                if (subSetSum == 0)
                    return true;
                if (n == 0 || subSetSum < 0)
                    return false;
                // check if subSetSum for given n is already computed and stored in memo
                if (memo[n][subSetSum] != null)
                    return memo[n][subSetSum];
                bool result = Dfs(nums, n - 1, subSetSum - nums[n - 1], memo) ||
                        Dfs(nums, n - 1, subSetSum, memo);
                // store the result in memo
                memo[n][subSetSum] = result;
                return result;
            }
            /* Approach 3: Bottom Up Dynamic Programming
Complexity Analysis
•	Time Complexity : O(m⋅n), where m is the subSetSum, and n is the number of array elements. We iteratively fill the array of size m⋅n.
•	Space Complexity : O(m⋅n) , where n is the number of array elements and m is the subSetSum. We are using a 2 dimensional array dp of size m⋅n

             */
            public bool BottomUpDP(int[] nums)
            {
                int totalSum = 0;
                // find sum of all array elements
                foreach (int num in nums)
                {
                    totalSum += num;
                }
                // if totalSum is odd, it cannot be partitioned into equal sum subset
                if (totalSum % 2 != 0) return false;
                int subSetSum = totalSum / 2;
                int n = nums.Length;
                bool[][] dp = new bool[n + 1][];
                dp[0][0] = true;
                for (int i = 1; i <= n; i++)
                {
                    int curr = nums[i - 1];
                    for (int j = 0; j <= subSetSum; j++)
                    {
                        if (j < curr)
                            dp[i][j] = dp[i - 1][j];
                        else
                            dp[i][j] = dp[i - 1][j] || (dp[i - 1][j - curr]);
                    }
                }
                return dp[n][subSetSum];
            }

            /* Approach 4: Optimised Dynamic Programming - Using 1D Array
Complexity Analysis
•	Time Complexity : O(m⋅n), where m is the subSetSum, and n is the number of array elements. The time complexity is the same as Approach 3.
•	Space Complexity: O(m), As we use an array of size m to store the result of subproblems

             */
            public bool BottomUp1DDP(int[] nums)
            {
                if (nums.Length == 0)
                    return false;
                int totalSum = 0;
                // find sum of all array elements
                foreach (int num in nums)
                {
                    totalSum += num;
                }
                // if totalSum is odd, it cannot be partitioned into equal sum subset
                if (totalSum % 2 != 0) return false;
                int subSetSum = totalSum / 2;
                bool[] dp = new bool[subSetSum + 1];
                dp[0] = true;
                foreach (int curr in nums)
                {
                    for (int j = subSetSum; j >= curr; j--)
                    {
                        dp[j] |= dp[j - curr];
                    }
                }
                return dp[subSetSum];
            }

        }

        /* 698. Partition to K Equal Sum Subsets
        https://leetcode.com/problems/partition-to-k-equal-sum-subsets/description/
         */
        class CanPartitionKSubsetsSol
        {
            /*             Approach 1: Naive Backtracking
            Complexity Analysis
            Let N denote the number of elements in the array.
            •	Time complexity: O(N⋅N!).
            The idea is that for each recursive call, we will iterate over N elements and make another recursive call. Assume we picked one element, then we iterate over the array and make recursive calls for the next N−1 elements and so on.
            Therefore, in the worst-case scenario, the total number of recursive calls will be N⋅(N−1)⋅(N−2)⋅...⋅2⋅1=N! and in each recursive call we perform an O(N) time operation.
            Another way is to visualize all possible states by drawing a recursion tree. From root node we have N recursive calls. The first level, therefore, has N nodes. For each of the nodes in the first level, we have (N−1) similar choices. As a result, the second level has N∗(N−1) nodes, and so on. The last level must have N⋅(N−1)⋅(N−2)⋅(N−3)⋅...⋅2⋅1 nodes.
            •	Space complexity: O(N).
            We have used an extra array of size N to mark the picked elements.
            And the maximum depth of the recursive tree is at most N, so the recursive stack also takes O(N) space.
             */
            public bool UsingNaiveBacktracking(int[] arr, int k)
            {
                int totalArraySum = 0;
                int n = arr.Length;

                for (int i = 0; i < n; ++i)
                {
                    totalArraySum += arr[i];
                }

                // If total sum not divisible by k, we can't make subsets.
                if (totalArraySum % k != 0)
                {
                    return false;
                }

                int targetSum = totalArraySum / k;
                bool[] taken = new bool[n];

                return Backtrack(arr, 0, 0, k, targetSum, taken);
            }
            private bool Backtrack(int[] arr, int count, int currSum, int k, int targetSum, bool[] taken)
            {
                int n = arr.Length;

                // We made k - 1 subsets with target sum and last subset will also have target sum.
                if (count == k - 1)
                {
                    return true;
                }

                // Current subset sum exceeds target sum, no need to proceed further.
                if (currSum > targetSum)
                {
                    return false;
                }

                // When current subset sum reaches target sum then one subset is made.
                // Increment count and reset current subset sum to 0.
                if (currSum == targetSum)
                {
                    return Backtrack(arr, count + 1, 0, k, targetSum, taken);
                }

                // Try not picked elements to make some combinations.
                for (int j = 0; j < n; ++j)
                {
                    if (!taken[j])
                    {
                        // Include this element in current subset.
                        taken[j] = true;

                        // If using current jth element in this subset leads to make all valid subsets.
                        if (Backtrack(arr, count, currSum + arr[j], k, targetSum, taken))
                        {
                            return true;
                        }

                        // Backtrack step.
                        taken[j] = false;
                    }
                }

                // We were not able to make a valid combination after picking each element from the array,
                // hence we can't make k subsets.
                return false;
            }
            /* Approach 2: Optimized Backtracking 
Complexity Analysis
Let N be the number of elements in the array and k be the number of subsets.
•	Time complexity: O(k⋅2^N).
We are traversing the entire array for each subset (once we are done with one subset, for the next subset we are starting again with index 0). So for each subset, we are choosing the suitable elements from the array (basically iterate over nums and for each element either use it or skip it, which is O(2^N) operation).
Once the first subset is found, we go on to find the second, which would take 2^N operations roughly (because some numbers have been marked as visited). So T=2^N+2^N+2^N+...=k⋅2^N.
Also, one point to note that some might think if we take 2N time in finding one subset and we find k such subsets, then it will lead to 2^N⋅2^N⋅...(ktimes)..⋅2^N=2(N⋅k) time complexity. However, this is a misconception, as we iterate over array k times to get k subsets, hence total time is k⋅(timetogetonesubset)=k⋅2^N.
•	Space complexity: O(N).
We have used an extra array of size N to mark the already used elements.
And the recursive tree makes at most N calls at one time, so the recursive stack also takes O(N) space.

            */
            public bool UsingOptimalBacktracking(int[] arr, int k)
            {
                int totalArraySum = 0;
                int n = arr.Length;

                for (int i = 0; i < n; ++i)
                {
                    totalArraySum += arr[i];
                }

                // If total sum not divisible by k, we can't make subsets.
                if (totalArraySum % k != 0)
                {
                    return false;
                }

                // Sort in decreasing order.
                Array.Sort(arr);
                Reverse(arr);

                int targetSum = totalArraySum / k;
                bool[] taken = new bool[n];

                return Backtrack(arr, 0, 0, 0, k, targetSum, taken);
            }
            private bool Backtrack(int[] arr, int index, int count, int currSum, int k,
                          int targetSum, bool[] taken)
            {

                int n = arr.Length;

                // We made k - 1 subsets with target sum and last subset will also have target sum.
                if (count == k - 1)
                {
                    return true;
                }

                // No need to proceed further.
                if (currSum > targetSum)
                {
                    return false;
                }

                // When curr sum reaches target then one subset is made.
                // Increment count and reset current sum.
                if (currSum == targetSum)
                {
                    return Backtrack(arr, 0, count + 1, 0, k, targetSum, taken);
                }

                // Try not picked elements to make some combinations.
                for (int j = index; j < n; ++j)
                {
                    if (!taken[j])
                    {
                        // Include this element in current subset.
                        taken[j] = true;

                        // If using current jth element in this subset leads to make all valid subsets.
                        if (Backtrack(arr, j + 1, count, currSum + arr[j], k, targetSum, taken))
                        {
                            return true;
                        }

                        // Backtrack step.
                        taken[j] = false;
                    }
                }

                // We were not able to make a valid combination after picking each element from the array,
                // hence we can't make k subsets.
                return false;
            }

            void Reverse(int[] arr)
            {
                for (int i = 0, j = arr.Length - 1; i < j; i++, j--)
                {
                    int temp = arr[i];
                    arr[i] = arr[j];
                    arr[j] = temp;
                }
            }


            /* Approach 3: Backtracking plus Memoization
Complexity Analysis
Let N denote the number of elements in the array.
•	Time complexity: O(N⋅2^N).
There will be 2^N unique combinations of the taken string, in which every combination of the given array will be linearly iterated. And if a combination occurs again then we just return the stored answer for it.


•	Space complexity: O(N⋅2^N).
There will be 2^N unique combinations of the taken string, and each string of size N will be stored in the map. Also, the recursive stack will use at most O(N) space at any time.

             */
            public bool UsingBacktrackingWithMemo(int[] array, int k)
            {
                int totalArraySum = 0;
                int n = array.Length;

                for (int i = 0; i < n; ++i)
                {
                    totalArraySum += array[i];
                }

                // If total sum not divisible by k, we can't make subsets.
                if (totalArraySum % k != 0)
                {
                    return false;
                }

                // Sort in decreasing order.
                Array.Sort(array);
                Reverse(array);

                int targetSum = totalArraySum / k;

                char[] taken = new char[n];
                for (int i = 0; i < n; ++i)
                {
                    taken[i] = '0';
                }

                // Memoize the answer using taken element's string as key.
                Dictionary<string, bool> memo = new Dictionary<string, bool>();

                return Backtrack(array, 0, 0, 0, k, targetSum, taken, memo);
            }
            private bool Backtrack(int[] array, int index, int count, int currentSum, int k,
                       int targetSum, char[] taken, Dictionary<string, bool> memo)
            {

                int n = array.Length;

                // We made k - 1 subsets with target sum and last subset will also have target sum.
                if (count == k - 1)
                {
                    return true;
                }

                // No need to proceed further.
                if (currentSum > targetSum)
                {
                    return false;
                }

                string takenStr = new string(taken);

                // If we have already computed the current combination.
                if (memo.ContainsKey(takenStr))
                {
                    return memo[takenStr];
                }

                // When current sum reaches target then one subset is made.
                // Increment count and reset current sum.
                if (currentSum == targetSum)
                {
                    bool answer = Backtrack(array, 0, count + 1, 0, k, targetSum, taken, memo);
                    memo[takenStr] = answer;
                    return answer;
                }

                // Try not picked elements to make some combinations.
                for (int j = index; j < n; ++j)
                {
                    if (taken[j] == '0')
                    {
                        // Include this element in current subset.
                        taken[j] = '1';

                        // If using current jth element in this subset leads to make all valid subsets.
                        if (Backtrack(array, j + 1, count, currentSum + array[j], k, targetSum, taken, memo))
                        {
                            return true;
                        }

                        // Backtrack step.
                        taken[j] = '0';
                    }
                }

                // We were not able to make a valid combination after picking each element from array,
                // hence we can't make k subsets.
                memo[takenStr] = false;
                return false;

            }
            /* Approach 4: Backtracking plus Memoization with Bitmasking
Complexity Analysis
Let N denote the number of elements in the array.
•	Time complexity: O(N⋅2^N).
There will be 2^N unique combinations of building a subset from the array of length N, in every combination the given array will be linearly iterated. And if a combination occurs again, then we just return the stored answer for it.


•	Space complexity: O(2^N).
There will be 2^N unique combinations of the integer mask which will be stored in the map. And recursive stack will require at most O(N) space.

             */
            public bool UsingBacktrackingWithMemoAndBitmasking(int[] arr, int k)
            {
                int totalArraySum = 0;
                int n = arr.Length;

                for (int i = 0; i < n; ++i)
                {
                    totalArraySum += arr[i];
                }

                // If total sum not divisible by k, we can't make subsets.
                if (totalArraySum % k != 0)
                {
                    return false;
                }

                // Sort in decreasing order.
                Array.Sort(arr);
                Reverse(arr);

                int targetSum = totalArraySum / k;
                int mask = 0;

                // Memoize the ans using taken element's string as key.
                Dictionary<int, bool> memo = new Dictionary<int, bool>();

                return Backtrack(arr, 0, 0, 0, k, targetSum, mask, memo);
            }
            private bool Backtrack(int[] arr, int index, int count, int currSum, int k,
                        int targetSum, int mask, Dictionary<int, bool> memo)
            {
                int n = arr.Length;

                // We made k - 1 subsets with target sum and last subset will also have target sum.
                if (count == k - 1)
                {
                    return true;
                }

                // No need to proceed further.
                if (currSum > targetSum)
                {
                    return false;
                }

                // If we have already computed the current combination.
                if (memo.ContainsKey(mask))
                {
                    return memo[mask];
                }

                // When curr sum reaches target then one subset is made.
                // Increment count and reset current sum.
                if (currSum == targetSum)
                {
                    bool ans = Backtrack(arr, 0, count + 1, 0, k, targetSum, mask, memo);
                    memo[mask] = ans;
                    return ans;
                }

                // Try not picked elements to make some combinations.
                for (int j = index; j < n; ++j)
                {
                    if (((mask >> j) & 1) == 0)
                    {
                        // Include this element in current subset.
                        mask |= (1 << j);

                        // If using current jth element in this subset leads to make all valid subsets.
                        if (Backtrack(arr, j + 1, count, currSum + arr[j], k, targetSum, mask, memo))
                        {
                            return true;
                        }

                        // Backtrack step.
                        mask ^= (1 << j);
                    }
                }

                // We were not able to make a valid combination after picking each element from the array,
                // hence we can't make k subsets.
                memo[mask] = false;
                return false;
            }
            /* Approach 5: Tabulation plus Bitmasking
Complexity Analysis
Assume N denotes the number of elements in the array.
•	Time complexity: O(N⋅2^N).
There will be 2^N unique states of the mask and in each state, we iterate over the whole array.


•	Space complexity: O(2^N).
There will be 2^N unique states of the mask which will be stored in the subsetSum array.

             */
            public bool UsingTabulationAndBitmasking(int[] arr, int k)
            {
                int totalArraySum = 0;
                int n = arr.Length;

                for (int i = 0; i < n; ++i)
                {
                    totalArraySum += arr[i];
                }

                // If total sum not divisible by k, we can't make subsets.
                if (totalArraySum % k != 0)
                {
                    return false;
                }

                int targetSum = totalArraySum / k;

                int[] subsetSum = new int[(1 << n)];
                for (int i = 0; i < (1 << n); ++i)
                {
                    subsetSum[i] = -1;
                }
                // Initially only one state is valid, i.e don't pick anything.
                subsetSum[0] = 0;

                for (int mask = 0; mask < (1 << n); mask++)
                {
                    // If the current state has not been reached earlier.
                    if (subsetSum[mask] == -1)
                    {
                        continue;
                    }

                    for (int i = 0; i < n; i++)
                    {
                        // If the number arr[i] was not picked earlier, and arr[i] + subsetSum[mask]
                        // is not greater than the targetSum then add arr[i] to the subset
                        // sum at subsetSum[mask] and store the result at subsetSum[mask | (1 << i)].
                        if ((mask & (1 << i)) == 0 && subsetSum[mask] + arr[i] <= targetSum)
                        {
                            subsetSum[mask | (1 << i)] = (subsetSum[mask] + arr[i]) % targetSum;
                        }
                    }

                    if (subsetSum[(1 << n) - 1] == 0)
                    {
                        return true;
                    }
                }

                return subsetSum[(1 << n) - 1] == 0;
            }



        }

        /* 1146. Snapshot Array
        https://leetcode.com/problems/snapshot-array/description/
         */
        class SnapshotArray
        {
            int snapId = 0;
            SortedDictionary<int, int>[] historyRecords;

            /* Approach: Binary Search
            Complexity Analysis
Let n be the maximum number of calls and k = length.
•	Time complexity: O(nlogn+k)
o	We initialize historyRecords with size k.
o	In the worst-case scenario, the number of calls to get, set, and snap are all O(n).
o	For each call to get(index, snap_id), we will perform a binary search over the list of records of nums[index]. Since a list contains at most O(n) records, a binary search takes O(logn) time on average. Thus it requires O(nlogn) time.
o	Each call to snap takes O(1) time.
o	Each call to set(index, snap_id) appends a pair to the historical record of nums[index], which takes O(1) time, or O(logn) in Java as we are using TreeMap.
•	Space complexity: O(n+k)
o	We initialize historyRecords with size k.
o	We add one pair (snap_id, val) for each call to set, thus there are at most n pairs saved in history_record.

             */
            public SnapshotArray(int length)
            {
                historyRecords = new SortedDictionary<int, int>[length];
                for (int i = 0; i < length; i++)
                {
                    historyRecords[i] = new SortedDictionary<int, int>();
                    historyRecords[i].Add(0, 0);
                }
            }

            public void Set(int index, int val)
            {
                historyRecords[index][snapId] = val;
            }

            public int Snap()
            {
                return snapId++;
            }

            public int Get(int index, int snapId)
            {
                return historyRecords[index].FloorEntry(snapId).Value;
            }
        }

        /* 451. Sort Characters By Frequency
        https://leetcode.com/problems/sort-characters-by-frequency/description/
         */
        public class FrequencySortSol
        {

            /* Approach 1: Arrays and Sorting	
            Complexity Analysis
Let n be the length of the input String.
•	Time Complexity : O(nlogn).
The first part of the algorithm, converting the String to a List of characters, has a cost of O(n), because we are adding n characters to the end of a List.
The second part of the algorithm, sorting the List of characters, has a cost of O(nlogn).
The third part of the algorithm, grouping the characters into Strings of similar characters, has a cost of O(n) because each character is being inserted once into a StringBuilder and once converted into a String.
Finally, the fourth part of the algorithm, sorting the Strings by length, has a worst case cost of O(n), which occurs when all the characters in the input String are unique.
Because we drop constants and insignificant terms, we get O(nlogn)+3⋅O(n)=O(nlogn).
Be careful with your own implementation—if you didn't do the string building process in a sensible way, then your solution could potentially be O(n2).
•	Space Complexity : O(n).
It is impossible to do better with the space complexity, because Strings are immutable. The List of characters, List of Strings, and the final output String, are all of length n, so we have a space complexity of O(n).

             */
            public string UsingArraysAndSort(string inputString)
            {
                if (string.IsNullOrEmpty(inputString)) return inputString;

                // Create a sorted array of characters.
                char[] characters = inputString.ToCharArray();
                Array.Sort(characters);

                // Convert identical characters into single strings.
                List<string> characterStrings = new List<string>();
                StringBuilder currentString = new StringBuilder();
                currentString.Append(characters[0]);
                for (int index = 1; index < characters.Length; index++)
                {
                    if (characters[index] != characters[index - 1])
                    {
                        characterStrings.Add(currentString.ToString());
                        currentString = new StringBuilder();
                    }
                    currentString.Append(characters[index]);
                }
                characterStrings.Add(currentString.ToString());

                // Our comparator is (a, b) => b.Length - a.Length.
                // If a is longer than b, then a negative number will be returned
                // telling the sort algorithm to place a first. Otherwise, a positive
                // number will be returned, telling it to place a second.
                // This results in a longest to shortest sorted list of the strings.
                characterStrings.Sort((a, b) => b.Length - a.Length);

                // Use StringBuilder to build the string to return.
                StringBuilder resultStringBuilder = new StringBuilder();
                foreach (string str in characterStrings) resultStringBuilder.Append(str);
                return resultStringBuilder.ToString();
            }


            /* Approach 2: HashMap/Dict and Sort
Complexity Analysis
Let n be the length of the input String and k be the number of unique characters in the String.
We know that k≤n, because there can't be more unique characters than there are characters in the String. We also know that k is somewhat bounded by the fact that there's only a finite number of characters in Unicode (or ASCII, which I suspect is all we need to worry about for this question).
There are two ways of approaching the complexity analysis for this question.
1.	We can disregard k, and consider that in the worst case, k = n.
2.	We can consider k, recognising that the number of unique characters is not infinite. This is more accurate for real world purposes.
I've provided analysis for both ways of approaching it. I choose not to bring it up for the previous approach, because it made no difference there.
•	Time Complexity : O(nlogn) OR O(n+klogk).
Putting the characterss into the HashMap has a cost of O(n), because each of the n characterss must be put in, and putting each in is an O(1) operation.
Sorting the HashMap keys has a cost of O(klogk), because there are k keys, and this is the standard cost for sorting. If only using n, then it's O(nlogn). For the previous question, the sort was carried out on n items, not k, so was possibly a lot worse.
Traversing over the sorted keys and building the String has a cost of O(n), as n characters must be inserted.
Therefore, if we're only considering n, then the final cost is O(nlogn).
Considering k as well gives us O(n+klogk), because we don't know which is largest out of n and klogk. We do, however, know that in total this is less than or equal to O(nlogn).
•	Space Complexity : O(n).
The HashMap uses O(k) space.
However, the StringBuilder at the end dominates the space complexity, pushing it up to O(n), as every character from the input String must go into it. Like was said above, it's impossible to do better with the space complexity here.
What's interesting here is that if we only consider n, the time complexity is the same as the previous approach. But by considering k, we can see that the difference is potentially substantial.

             */
            public string UsingDictAndSort(string inputString)
            {
                // Count up the occurrences.
                Dictionary<char, int> characterCounts = new Dictionary<char, int>();
                foreach (char character in inputString.ToCharArray())
                {
                    if (characterCounts.ContainsKey(character))
                    {
                        characterCounts[character]++;
                    }
                    else
                    {
                        characterCounts[character] = 1;
                    }
                }

                // Make a list of the keys, sorted by frequency.
                List<char> characters = characterCounts.Keys.ToList();
                characters.Sort((a, b) => characterCounts[b].CompareTo(characterCounts[a]));

                // Convert the counts into a string with a StringBuilder.
                StringBuilder stringBuilder = new StringBuilder();
                foreach (char character in characters)
                {
                    int copies = characterCounts[character];
                    for (int i = 0; i < copies; i++)
                    {
                        stringBuilder.Append(character);
                    }
                }
                return stringBuilder.ToString();
            }
            /* Approach 3: Multiset and Bucket Sort
Complexity Analysis
Let n be the length of the input String. The k (number of unique characters in the input String that we considered for the last approach makes no difference this time).
•	Time Complexity : O(n).
Like before, the HashMap building has a cost of O(n).
The bucket sorting is O(n), because inserting items has a cost of O(k) (each entry from the HashMap), and building the buckets initially has a worst case of O(n) (which occurs when k=1). Because k≤n, we're left with O(n).
So in total, we have O(n).
It'd be impossible to do better than this, because we need to look at each of the n characters in the input String at least once.
•	Space Complexity : O(n).
Same as above. The bucket Array also uses O(n) space, because its length is at most n, and there are k items across all the buckets.

             */
            public string UsingMultiSetAndBucketSort(string inputString)
            {

                if (string.IsNullOrEmpty(inputString)) return inputString;

                // Count up the occurrences.
                Dictionary<char, int> characterCounts = new Dictionary<char, int>();
                foreach (char character in inputString)
                {
                    if (characterCounts.ContainsKey(character))
                    {
                        characterCounts[character]++;
                    }
                    else
                    {
                        characterCounts[character] = 1;
                    }
                }

                int maximumFrequency = characterCounts.Values.Max();

                // Make the list of buckets and apply bucket sort.
                List<List<char>> buckets = new List<List<char>>();
                for (int i = 0; i <= maximumFrequency; i++)
                {
                    buckets.Add(new List<char>());
                }
                foreach (char key in characterCounts.Keys)
                {
                    int frequency = characterCounts[key];
                    buckets[frequency].Add(key);
                }

                // Build up the string. 
                StringBuilder stringBuilder = new StringBuilder();
                for (int i = buckets.Count - 1; i >= 1; i--)
                {
                    foreach (char character in buckets[i])
                    {
                        for (int j = 0; j < i; j++)
                        {
                            stringBuilder.Append(character);
                        }
                    }
                }
                return stringBuilder.ToString();
            }
        }

        /* 442. Find All Duplicates in an Array	
        https://leetcode.com/problems/find-all-duplicates-in-an-array/description/
         */
        class FindAllDuplicatesInArraySol
        {
            /* Approach 1: Brute Force 
            Complexity Analysis
            •	Time complexity: O(n^2).
            For each element in the array, we search for another occurrence in the rest of the array. Hence, for the ith element in the array, we might end up looking through all n−i remaining elements in the worst case. So, we can end up going through about n^2 elements in the worst case.
            n−1+n−2+n−3+....+1+0 = ∑1n(n−i) ≃ n^2
            •	Space complexity: O(1)
            No extra space is required, other than the space for the output list.
            */
            public List<int> Naive(int[] nums)
            {
                List<int> ans = new();

                for (int i = 0; i < nums.Length; i++)
                {
                    for (int j = i + 1; j < nums.Length; j++)
                    {
                        if (nums[j] == nums[i])
                        {
                            ans.Add(nums[i]);
                            break;
                        }
                    }
                }

                return ans;
            }
            /* Approach 2: Sort and Compare Adjacent Elements 
            Complexity Analysis
•	Time complexity: O(nlogn)+O(n)≃O(nlogn).
o	A performant comparison-based sorting algorithm will run in O(nlogn) time. Note that this can be reduced to O(n) using a special sorting algorithm like Radix Sort.
o	Traversing the array after sorting takes linear time i.e. O(n).
•	Space complexity: O(logn)
Note that some extra space is used when we sort an array in place. The space complexity of the sorting algorithm depends on the programming language.
o	In Java, Arrays.sort() is implemented using a variant of the Quick Sort algorithm which has a space complexity of O(logn) for sorting an array.
o	In C++, the sort() function is implemented as a hybrid of Quick Sort, Heap Sort, and Insertion Sort, with a worse-case space complexity of O(logn).
The space used by the output list ans is not counted in the space complexity.

            */
            public List<int> UsingSortAndCompareAdjacentElements(int[] nums)
            {
                List<int> ans = new();

                Array.Sort(nums);

                for (int i = 1; i < nums.Length; i++)
                {
                    if (nums[i] == nums[i - 1])
                    {
                        ans.Add(nums[i]);
                        i++;        // skip over next element
                    }
                }

                return ans;
            }
            /* Approach 3: Store Seen Elements in a Set / Map
            Complexity Analysis
•	Time complexity: O(n) average case. O(n) worst case.
o	It takes a linear amount of time to iterate through the array.
o	Lookups in a hash set are constant time on average, however, those can degrade to linear time in the worst case. Note that an alternative is to use tree-based sets, which give logarithmic time lookups always.
•	Space complexity: Up to O(n) extra space required for the set.
o	If you are tight on space, you can significantly reduce your physical space requirements by using bitsets 1 instead of sets. This data structure requires just one bit per element, so you can be done in just n bits of data for elements that go up to n. Of course, this doesn't reduce your space complexity: bitsets still grow linearly with the range of values that the elements can take.

             */
            public List<int> UsingStoreSeenElementsInASet(int[] nums)
            {
                List<int> ans = new();
                HashSet<int> seen = new();

                foreach (int num in nums)
                {
                    if (seen.Contains(num))
                    {
                        ans.Add(num);
                    }
                    else
                    {
                        seen.Add(num);
                    }
                }

                return ans;
            }
            /* Approach 4: Cycle Sort
                Complexity Analysis
Let n be the length of nums.
•	Time complexity: O(n)
We loop through the elements in nums, swapping elements to sort the array. Swapping takes constant time. Sorting nums using cycle sort takes O(n) time.
Iterating through the sorted array and finding the duplicates can take up to O(n).
The total time complexity is O(2n), which simplifies to O(n).
•	Space complexity: O(n)
We modify the array nums and use it to determine the answer, so the space complexity is O(n).
The space used by the output list duplicates is not counted in the space complexity.
nums is the input array, so the auxiliary space used is O(1).

             */
            public List<int> UsingCycleSort(int[] nums)
            {
                int n = nums.Length;

                // Use cycle sort to place elements
                // at corresponding index to value
                int i = 0;
                while (i < n)
                {
                    int correctIdx = nums[i] - 1;
                    if (nums[i] != nums[correctIdx])
                    {
                        Swap(nums, i, correctIdx);
                    }
                    else
                    {
                        i++;
                    }
                }

                // Any elements not at the index that corresponds to their value are duplicates
                List<int> duplicates = new();
                for (i = 0; i < n; i++)
                {
                    if (nums[i] != i + 1)
                    {
                        duplicates.Add(nums[i]);
                    }
                }

                return duplicates;
            }

            // Swaps two elements in nums
            private void Swap(int[] nums, int index1, int index2)
            {
                int temp = nums[index1];
                nums[index1] = nums[index2];
                nums[index2] = temp;
            }
            /*             
            Approach 5: Mark Visited Elements in the Input Array Itself
Complexity Analysis
•	Time complexity: O(n)
We iterate over the array twice. Each negation operation occurs in constant time.
•	Space complexity: O(n)
We modify the array nums and use it to determine the answer, so the space complexity is O(n).
The space used by the output list ans is not counted in the space complexity.
nums is the input array, so the auxiliary space used is O(1).


             */
            public List<int> UsingMarkVisitedElementsInInputArrayItself(int[] nums)
            {
                List<int> ans = new();

                foreach (int num in nums)
                {
                    nums[Math.Abs(num) - 1] *= -1;
                }

                foreach (int num in nums)
                {
                    if (nums[Math.Abs(num) - 1] > 0)
                    {
                        ans.Add(Math.Abs(num));
                        nums[Math.Abs(num) - 1] *= -1;
                    }
                }

                return ans;
            }

        }


        /* 3152. Special Array II
        https://leetcode.com/problems/special-array-ii/description/
         */
        class SpecialArrayIISol
        {
            /*             Complexity
            Time complexity:
            O(n + q)
            Space complexity:
            O(n)

             */
            public bool[] isArraySpecial(int[] nums, int[][] queries)
            {
                bool[] ans = new bool[queries.Length];
                int[] ps = new int[nums.Length];
                for (int i = 1; i < nums.Length; i++)
                {
                    ps[i] = ps[i - 1];
                    if (nums[i - 1] % 2 == nums[i] % 2)
                    {
                        ps[i]++;
                    }
                }
                for (int i = 0; i < queries.Length; i++)
                {
                    int from = queries[i][0], to = queries[i][1];
                    int cnt = ps[to] - ps[from];
                    ans[i] = cnt > 0 ? false : true;
                }
                return ans;
            }
        }


        /* 950. Reveal Cards In Increasing Order
        https://leetcode.com/problems/reveal-cards-in-increasing-order/description/
         */
        class DeckRevealedIncreasingSol
        {
            /* 
            Approach 1: Two Pointers
            Complexity Analysis
Let n be the length of the deck.
•	Time complexity: O(nlogn)
Sorting the deck takes O(nlogn).
The loop to place cards at the correct index in result runs O(nlogn) times. Each pass through the result array takes O(n), and with each pass, half as many indices still need to be filled.
Therefore, the overall time complexity is O(nlogn)
•	Space complexity: O(n) or O(logn).
result is only used to store the result, so it is not counted in the space complexity.
Some extra space is used when we sort the deck in place. The space complexity of the sorting algorithms depends on the programming language.
o	In Python, the sort method sorts a list using the Timesort algorithm which is a combination of Merge Sort and Insertion Sort and has O(n) additional space.
o	In C++, the sort() function is implemented as a hybrid of Quick Sort, Heap Sort, and Insertion Sort, with a worse-case space complexity of O(logn).
o	In Java, Arrays.sort() is implemented using a variant of the Quick Sort algorithm which has a space complexity of O(logn) for sorting two arrays.


             */
            public int[] UsingTwoPointers(int[] deck)
            {
                int N = deck.Length;
                int[] result = new int[N];
                bool skip = false;
                int indexInDeck = 0;
                int indexInResult = 0;

                Array.Sort(deck);

                while (indexInDeck < N)
                {
                    // There is an available gap in result
                    if (result[indexInResult] == 0)
                    {

                        // Add a card to result
                        if (!skip)
                        {
                            result[indexInResult] = deck[indexInDeck];
                            indexInDeck++;
                        }

                        // Toggle skip to alternate between adding and skipping cards
                        skip = !skip;
                    }
                    // Progress to next index of result array
                    indexInResult = (indexInResult + 1) % N;
                }
                return result;
            }

            /* Approach 2: Simulation with Queue
Complexity Analysis
Let n be the length of the deck.
•	Time complexity: O(nlogn)
Sorting the deck takes O(nlogn).
It takes O(n) time to build the queue. Then, it takes O(n) time to add the cards to the result array in the correct order.
The time used for sorting is the dominating term, so the overall time complexity is O(nlogn)
•	Space complexity: O(n)
We use a queue of size n, so the space complexity is O(n).
Some extra space is used when we sort the deck in place. The space complexity of the sorting algorithms depends on the programming language.
o	In Python, the sort method sorts a list using the Timesort algorithm which is a combination of Merge Sort and Insertion Sort and has O(n) additional space.
o	In C++, the sort() function is implemented as a hybrid of Quick Sort, Heap Sort, and Insertion Sort, with a worse-case space complexity of O(logn).
o	In Java, Arrays.sort() is implemented using a variant of the Quick Sort algorithm which has a space complexity of O(logn) for sorting two arrays.
As the dominating term is O(n), the overall space complexity is O(n).

             */
            public int[] UsingSimulationWithQueue(int[] deck)
            {
                int N = deck.Length;
                Queue<int> queue = new();

                // Create a queue of indexes
                for (int i = 0; i < N; i++)
                {
                    queue.Enqueue(i);
                }

                Array.Sort(deck);

                // Put cards at correct index in result
                int[] result = new int[N];
                for (int i = 0; i < N; i++)
                {
                    // Reveal Card and place in result
                    result[queue.Dequeue()] = deck[i];

                    // Move next card to bottom
                    queue.Enqueue(queue.Dequeue());
                }
                return result;
            }
        }

        /* 2149. Rearrange Array Elements by Sign
        https://leetcode.com/problems/rearrange-array-elements-by-sign/description/
         */

        class RearrangeArrayElenebtsBySignSol
        {
            /*Approach: Two Pointers

            Complexity Analysis
Let n be the length of nums

•	Time complexity: O(n)
o	We traverse nums once and populate ans. Since both these arrays have size n, this results in a time complexity of O(n).

•	Space complexity: O(n)
o	We create an auxiliary array ans of size n.

 */
            public int[] UsingTwoPointers(int[] nums)
            {
                int n = nums.Length;

                // Initializing an answer array of size n
                int[] ans = new int[n];

                // Initializing two pointers to track even and 
                // odd indices for positive and negative integers respectively
                int posIndex = 0, negIndex = 1;

                for (int i = 0; i < n; i++)
                {
                    if (nums[i] > 0)
                    {
                        // Placing the positive integer at the 
                        // desired index in ans and incrementing posIndex by 2
                        ans[posIndex] = nums[i];
                        posIndex += 2;
                    }
                    else
                    {
                        // Placing the negative integer at the 
                        // desired index in ans and incrementing negIndex by 2
                        ans[negIndex] = nums[i];
                        negIndex += 2;
                    }
                }

                return ans;
            }
        }


        /* 532. K-diff Pairs in an Array
        https://leetcode.com/problems/k-diff-pairs-in-an-array/description/
         */
        public class KDiffPairsInArraySol
        {

            /* Approach 1: Brute Force 
Complexity Analysis
•	Time complexity : O(N^2) where N is the size of nums. The time complexity for sorting is O(NlogN) while the time complexity for going through ever pair in the nums is O(N^2). Therefore, the final time complexity is O(NlogN)+O(N^2)≈O(N^2).
•	Space complexity : O(N) where N is the size of nums. This space complexity is incurred by the sorting algorithm. Space complexity is bound to change depending on the sorting algorithm you use. There is no additional space required for the part with two for loops, apart from a single variable result. Therefore, the final space complexity is O(N)+O(1)≈O(N).
Addendum: We can also approach this problem using brute force without sorting nums. First, we have to create a hash set which will record pairs of numbers whose difference is k. Then, we look for every possible pair. As soon as we find a pair (say i and j) whose difference is k, we add (i, j) and (j, i) to the hash set and increment our placeholder result variable. Whenever we encounter another pair which is already in the hash set, we simply ignore that pair. By doing so we have a better practical runtime (since we are eliminating the sorting step) even though the time complexity is still O(N^2) where N is the size of nums.

            */
            public int Naive(int[] nums, int k)
            {

                Array.Sort(nums);

                int result = 0;

                for (int i = 0; i < nums.Length; i++)
                {
                    if (i > 0 && nums[i] == nums[i - 1])
                        continue;
                    for (int j = i + 1; j < nums.Length; j++)
                    {
                        if (j > i + 1 && nums[j] == nums[j - 1])
                            continue;

                        if (Math.Abs(nums[j] - nums[i]) == k)
                            result++;
                    }
                }

                return result;
            }
            /* Approach 2: Two Pointers
            Complexity Analysis
•	Time complexity : O(NlogN) where N is the size of nums. The time complexity for sorting is O(NlogN) while the time complexity for going through nums is O(N). One might mistakenly think that it should be O(N^2) since there is another while loop inside the first while loop. The while loop inside is just incrementing the pointer to skip numbers which are the same as the previous number. The animation should explain this behavior clearer. Therefore, the final time complexity is O(NlogN)+O(N)≈O(NlogN).
•	Space complexity : O(N) where N is the size of nums. Similar to approach 1, this space complexity is incurred by the sorting algorithm. Space complexity is bound to change depending on the sorting algorithm you use. There is no additional space required for the part where two pointers are being incremented, apart from a single variable result. Therefore, the final space complexity is O(N)+O(1)≈O(N).

             */
            public int UsingTwoPointers(int[] nums, int k)
            {

                Array.Sort(nums);

                int left = 0, right = 1;
                int result = 0;

                while (left < nums.Length && right < nums.Length)
                {
                    if (left == right || nums[right] - nums[left] < k)
                    {
                        // List item 1 in the text
                        right++;
                    }
                    else if (nums[right] - nums[left] > k)
                    {
                        // List item 2 in the text
                        left++;
                    }
                    else
                    {
                        // List item 3 in the text
                        left++;
                        result++;
                        while (left < nums.Length && nums[left] == nums[left - 1])
                            left++;
                    }
                }
                return result;
            }

            /* Approach 3: Hashmap
Complexity Analysis
Let N be the number of elements in the input list.
•	Time complexity : O(N).
o	It takes O(N) to create an initial frequency hash map and another O(N) to traverse the keys of that hash map. One thing to note about is the hash key lookup. The time complexity for hash key lookup is O(1) but if there are hash key collisions, the time complexity will become O(N). However those cases are rare and thus, the amortized time complexity is O(2N)≈O(N).
•	Space complexity : O(N)
o	We keep a table to count the frequency of each unique number in the input. In the worst case, all numbers are unique in the array.
As a result, the maximum size of our table would be O(N).

             */
            public int FindPairs(int[] numbers, int difference)
            {
                int resultCount = 0;

                Dictionary<int, int> frequencyCounter = new Dictionary<int, int>();
                foreach (int number in numbers)
                {
                    if (frequencyCounter.ContainsKey(number))
                    {
                        frequencyCounter[number]++;
                    }
                    else
                    {
                        frequencyCounter[number] = 1;
                    }
                }

                foreach (KeyValuePair<int, int> entry in frequencyCounter)
                {
                    int currentNumber = entry.Key;
                    int frequency = entry.Value;
                    if (difference > 0 && frequencyCounter.ContainsKey(currentNumber + difference))
                    {
                        resultCount++;
                    }
                    else if (difference == 0 && frequency > 1)
                    {
                        resultCount++;
                    }
                }
                return resultCount;
            }
        }

        /* 2918. Minimum Equal Sum of Two Arrays After Replacing Zeros
        https://leetcode.com/problems/minimum-equal-sum-of-two-arrays-after-replacing-zeros/description/
        https://algo.monster/liteproblems/2918
         */

        class MinEqualSumOfTwoArraysAfterReplacingZerosSol
        {

            /* Time and Space Complexity
            Time Complexity
            The time complexity of the code can be broken down as follows:
            •	Calculating sum of nums1 and nums2 requires traversing both arrays, which costs O(n) and O(m) respectively, where n is the length of nums1 and m is the length of nums2.
            •	Counting the zeroes in nums1 and nums2 with nums1.count(0) and nums2.count(0) also takes O(n) and O(m) time respectively.
            •	The recursive call self.minSum(nums2, nums1) only happens if s1 > s2. If multiple invocations occur, the process will still not exceed O(n + m) because the sum and count operations are repeated once for each list.
            •	The if comparisons are constant time operations, O(1).
            Aggregating all these costs, the total time complexity is O(n + m) because we are summing separate linear terms associated with each list.
            Space Complexity
            •	The space complexity is O(1) as no additional space that grows with the input size is used. Variables s1 and s2 use constant space, and the recursive call does not exceed a single level of stack space since it's a direct swap with no further recursion.
             */
            /**
             * Computes the minimum sum possible between two given arrays after making sure
             * that none of the elements in the arrays are less than one.
             * It returns -1 if it is impossible to make both sums equal without the use of zeros.
             *
             * @param nums1 the first array of integers
             * @param nums2 the second array of integers
             * @return the minimum sum possible or -1 if it can't be done
             */
            public long MinSum(int[] nums1, int[] nums2)
            {
                long sum1 = 0; // Initialize sum for the first array
                long sum2 = 0; // Initialize sum for the second array
                bool hasZero = false; // Flag to check for presence of zero in nums1

                // Iterate over the first array
                foreach (int value in nums1)
                {
                    hasZero |= value == 0; // Set the flag if zero is found
                    sum1 += Math.Max(value, 1); // Ensure that each value contributes at least 1 to the sum
                }

                // Iterate over the second array
                foreach (int value in nums2)
                {
                    sum2 += Math.Max(value, 1); // Ensure that each value contributes at least 1 to the sum
                }

                // If the sum of the first array is greater, call the function again with reversed parameters
                if (sum1 > sum2)
                {
                    return MinSum(nums2, nums1);
                }

                // If the sums are equal, return the sum of the first array
                if (sum1 == sum2)
                {
                    return sum1;
                }

                // If there is a zero in the first array and the sums are not equal, returning the sum of the
                // second array is valid; otherwise, return -1 as it is impossible to make sums equal.
                return hasZero ? sum2 : -1;
            }
        }

        /* 2592. Maximize Greatness of an Array
        https://leetcode.com/problems/maximize-greatness-of-an-array/description/
        https://algo.monster/liteproblems/2592
         */

        class MaximizeGreatnessSol
        {
            /* Time and Space Complexity
            Time Complexity
            The time complexity of the given function maximizeGreatness is primarily dominated by the nums.sort() operation. Sorting an array is typically done using the Timsort algorithm in Python, which has a time complexity of O(n log n), where n is the length of the array nums. The for-loop that follows the sort operation has a time complexity of O(n) since it iterates over each element in the sorted list exactly once.
            Therefore, when combining these two operations, the total time complexity remains O(n log n) as the sorting operation is the most significant factor.
            Space Complexity
            The space complexity of the sort operation in Python is O(n), but this uses the internal buffer for sorting which does not count towards the additional space required by the algorithm. However, due to the recursive stack calls made during sorting, the worst-case space complexity is O(log n). No additional data structures that are dependent on the size of the input array are used in the function, keeping the space complexity to O(log n) for this function.
             */
            // Method to determine the maximum level of greatness that can be achieved
            public int MaximizeGreatness(int[] nums)
            {
                // Sort the array in non-decreasing order
                Array.Sort(nums);

                // Initialize the count of greatness to 0
                int greatnessCount = 0;

                // Iterate over the sorted array
                foreach (int num in nums)
                {
                    // If the current element is greater than the element at the current greatness count index
                    if (num > nums[greatnessCount])
                    {
                        // Increment the greatness count
                        greatnessCount++;
                    }
                }

                // Return the total count of greatness achieved
                return greatnessCount;
            }
        }

        /* 526. Beautiful Arrangement
        https://leetcode.com/problems/beautiful-arrangement/description/
         */

        public class CountArrangementSol
        {
            int count = 0;
            /* Approach #1 Brute Force [Time Limit Exceeded]	 
            Complexity Analysis
•	Time complexity : O(n!). A total of n! permutations will be generated for an array of length n.
•	Space complexity : O(n). The depth of the recursion tree can go upto n. nums array of size n is used.

            */
            public int Naive(int N)
            {
                int[] nums = new int[N];
                for (int i = 1; i <= N; i++)
                    nums[i - 1] = i;
                Permute(nums, 0);
                return count;
            }
            public void Permute(int[] nums, int l)
            {
                if (l == nums.Length - 1)
                {
                    int i;
                    for (i = 1; i <= nums.Length; i++)
                    {
                        if (nums[i - 1] % i != 0 && i % nums[i - 1] != 0)
                            break;
                    }
                    if (i == nums.Length + 1)
                    {
                        count++;
                    }
                }
                for (int i = l; i < nums.Length; i++)
                {
                    Swap(nums, i, l);
                    Permute(nums, l + 1);
                    Swap(nums, i, l);
                }
            }
            public void Swap(int[] nums, int x, int y)
            {
                int temp = nums[x];
                nums[x] = nums[y];
                nums[y] = temp;
            }


            /* Approach #2 Better Brute Force [Accepted]
Complexity Analysis
•	Time complexity : O(k). k refers to the number of valid permutations.
•	Space complexity : O(n). The depth of recursion tree can go upto n. Further, nums array of size n is used, where, n is the given number.

             */
            public int countArrangement(int N)
            {
                int[] nums = new int[N];
                for (int i = 1; i <= N; i++)
                    nums[i - 1] = i;
                Permute(nums, 0);
                return count;

                void Permute(int[] nums, int l)
                {
                    if (l == nums.Length)
                    {
                        count++;
                    }
                    for (int i = l; i < nums.Length; i++)
                    {
                        Swap(nums, i, l);
                        if (nums[l] % (l + 1) == 0 || (l + 1) % nums[l] == 0)
                            Permute(nums, l + 1);
                        Swap(nums, i, l);
                    }
                }
            }
            /* Approach #3 Backtracking [Accepted]
            **Complexity Analysis**
•	Time complexity : O(k). k refers to the number of valid permutations.
•	Space complexity : O(n). visited array of size n is used. The depth of recursion tree will also go upto n. Here, n refers to the given integer n.

             */
            public int UsingBacktracking(int N)
            {
                bool[] visited = new bool[N + 1];
                Calculate(N, 1, visited);
                return count;
            }
            public void Calculate(int N, int pos, bool[] visited)
            {
                if (pos > N)
                    count++;
                for (int i = 1; i <= N; i++)
                {
                    if (!visited[i] && (pos % i == 0 || i % pos == 0))
                    {
                        visited[i] = true;
                        Calculate(N, pos + 1, visited);
                        visited[i] = false;
                    }
                }
            }

        }

        /* 713. Subarray Product Less Than K
        https://leetcode.com/problems/subarray-product-less-than-k/description/
         */

        class NumSubarrayProductLessThanKSol
        {

            /* Approach 1: Using Sliding Window
Complexity Analysis
Let n be the length of nums.
•	Time complexity: O(n)
The algorithm iterates through the input array nums using a single for loop. Inside the loop, there are nested operations for shrinking the window, but since left is incremented a total number of n times during the whole array traversal, each element in the array is visited at most twice.
The nested loop terminates when the product becomes less than k, and this can only happen at most n times total (once for each element). Therefore, the overall time complexity is 2n, which we describe as O(n).
•	Space complexity: O(1)
The algorithm uses a constant amount of extra space for variables like totalCount, product, left, and right. These variables do not depend on the size of the input array. Therefore, the space complexity is considered constant, denoted as O(1).
________________________________________
Note: The below approach is generally not anticipated in an interview setting, as many individuals might not be familiar with logarithmic functions, having either forgotten them or not utilized them extensively. So, it's tough for them to come up with this idea on the spot. Moreover, the sliding window approach remains the optimal solution to this problem.

             */
            public int NumSubarrayProductLessThanK(int[] nums, int k)
            {
                // Handle edge cases where k is 0 or 1 (no subarrays possible)
                if (k <= 1) return 0;

                int totalCount = 0;
                int product = 1;

                // Use two pointers to maintain a sliding window
                for (int left = 0, right = 0; right < nums.Length; right++)
                {
                    // Expand the window by including the element at the right pointer
                    product *= nums[right];

                    // Shrink the window from the left while the product is greater than or equal to k
                    while (product >= k)
                    {
                        // Remove the element at the left pointer from the product
                        product /= nums[left++];
                    }

                    // Update the total count by adding the number of valid subarrays with the current window size
                    totalCount += right - left + 1;  // right - left + 1 represents the current window size
                }

                return totalCount;
            }

            /* Approach 2: Using Binary Search
            Complexity Analysis
Let n be the length of the nums array.
•	Time complexity: O(n⋅log(n))
The time complexity of the overall algorithm is O(n⋅log(n)) due to the binary search performed in each iteration of the outer loop.
•	Space complexity: O(n)
The space complexity is O(n) due to the list logsPrefixSum, storing logarithmic prefix sums of nums, whose length equals that of nums.

            */
            public int numSubarrayProductLessThanK(int[] nums, int k)
            {
                if (k == 0) return 0;
                double logK = Math.Log(k);
                int m = nums.Length + 1;
                double[] logsPrefixSum = new double[m];

                // Calculate prefix sum of logarithms of elements
                for (int i = 0; i < nums.Length; i++)
                {
                    logsPrefixSum[i + 1] = logsPrefixSum[i] + Math.Log(nums[i]);
                }

                int totalCount = 0;
                // Calculate subarray count with product less than k
                for (int currIdx = 0; currIdx < m; currIdx++)
                {
                    int low = currIdx + 1, high = m;
                    while (low < high)
                    {
                        int mid = low + (high - low) / 2;
                        if (logsPrefixSum[mid] < logsPrefixSum[currIdx] + logK - 1e-9)
                        {
                            low = mid + 1;
                        }
                        else
                        {
                            high = mid;
                        }
                    }
                    totalCount += low - currIdx - 1;
                }
                return totalCount;
            }
        }

        /* 
        357. Count Numbers with Unique Digits
        https://leetcode.com/problems/count-numbers-with-unique-digits/description/
        https://algo.monster/liteproblems/357
         */
        class CountNumbersWithUniqueDigitsSol
        {

            /* Time and Space Complexity
            The provided Python code defines a function countNumbersWithUniqueDigits which calculates the number of n-digit integers that have unique digits.
            •	Time Complexity: The time complexity of the function is primarily determined by the for loop that iterates n - 1 times. Within the for loop, there are only constant-time operations. Therefore, the overall time complexity is O(n).
            •	Space Complexity: The space complexity of the function is O(1) because the space used does not grow with the input size n. The function only uses a constant amount of additional space for variables ans and cur.
             */
            // This method counts the numbers with unique digits up to a certain length.
            public int CountNumbersWithUniqueDigits(int n)
            {
                // If n is 0, there's only one number which is 0 itself
                if (n == 0)
                {
                    return 1;
                }

                // If n is 1, we have digits from 0 to 9, resulting in 10 unique numbers
                if (n == 1)
                {
                    return 10;
                }

                // Initialize answer with the count for n = 1
                int answer = 10;

                // Current number of unique digits as we increase the length
                int currentUniqueNumbers = 9;

                // Loop to calculate the number of unique digit numbers for lengths 2 to n
                for (int i = 0; i < n - 1; ++i)
                {
                    // Compute the count for the current length by multiplying with the digits
                    // available considering we can't reuse any we have already used
                    currentUniqueNumbers *= (9 - i);

                    // Add the current length's count to the total answer so far
                    answer += currentUniqueNumbers;
                }

                // Return the total count of unique numbers with digits up to length n
                return answer;
            }
        }

        /* 2563. Count the Number of Fair Pairs
        https://leetcode.com/problems/count-the-number-of-fair-pairs/description/
        https://algo.monster/liteproblems/2563
         */
        class CountFairPairsSol
        {
            /* Time and Space Complexity
            Time Complexity
            The given Python code performs the sorting of the nums list, which takes O(n log n) time, where n is the number of elements in the list. After sorting, it iterates over each element in nums and performs two binary searches using the bisect_left function.
            For each element x in the list, it finds the index j of the first number not less than lower - x starting from index i + 1 and the index k of the first number not less than upper - x + 1 from the same index i + 1. The binary searches take O(log n) time each.
            Since the binary searches are inside a loop that runs n times, the total time for all binary searches combined is O(n log n). This means the overall time complexity of the function is dominated by the sorting and binary searches, which results in O(n log n).
            Space Complexity
            The space complexity of the algorithm is O(1) if we disregard the input and only consider additional space because the sorting is done in-place and the only other variables are used for iteration and counting.
            In the case where the sorting is not done in-place (depending on the Python implementation), the space complexity would be O(n) due to the space needed to create a sorted copy of the list. However, typically, the .sort() method on a list in Python sorts in-place, thus the typical space complexity remains O(1).
             */
            // Counts the number of 'fair' pairs in the array, where a pair is considered fair 
            // if the sum of its elements is between 'lower' and 'upper' (inclusive).
            public long CountFairPairs(int[] nums, int lower, int upper)
            {
                // Sort the array to enable binary search
                Array.Sort(nums);
                long count = 0; // Initialize count of fair pairs
                int n = nums.Length;

                // Iterate over each element in the array
                for (int i = 0; i < n; ++i)
                {
                    // Find the left boundary for the fair sum range
                    int leftBoundaryIndex = BinarySearch(nums, lower - nums[i], i + 1);

                    // Find the right boundary for the fair sum range
                    int rightBoundaryIndex = BinarySearch(nums, upper - nums[i] + 1, i + 1);

                    // Calculate the number of fair pairs with the current element
                    count += rightBoundaryIndex - leftBoundaryIndex;
                }

                // Return the total count of fair pairs
                return count;
            }

            // Performs a binary search to find the index of the smallest number in 'nums' 
            // starting from 'startIdx' that is greater or equal to 'target'.
            private int BinarySearch(int[] nums, int target, int startIdx)
            {
                int endIdx = nums.Length; // Sets the end index of the search range

                // Continue the loop until the search range is exhausted
                while (startIdx < endIdx)
                {
                    int midIdx = (startIdx + endIdx) >> 1; // Calculate the mid index
                                                           // If the mid element is greater or equal to target,
                                                           // we need to continue in the left part of the array
                    if (nums[midIdx] >= target)
                    {
                        endIdx = midIdx;
                    }
                    else
                    {
                        // Otherwise, continue in the right part
                        startIdx = midIdx + 1;
                    }
                }

                // Return the start index which is the index of the smallest
                // number greater or equal to 'target'.
                return startIdx;
            }
        }

        /* 1802. Maximum Value at a Given Index in a Bounded Array
        https://leetcode.com/problems/maximum-value-at-a-given-index-in-a-bounded-array/description/
         */
        class MaxValueAtGivenIndexInABoundedArraySol
        {
            /* Approach: Greedy + Binary Search
Complexity Analysis
•	Time complexity: O(log(maxSum))
o	We set the searching space as [1, maxSum], thus it takes O(log(maxSum)) steps to finish the binary search.
o	At each step, we made some calculations that take O(1) time.
•	Space complexity: O(1)
o	Both the binary search and the getSum function take O(1) space.

             */

            public int UsingGreedyAndBinarySearch(int n, int index, int maxSum)
            {
                int left = 1, right = maxSum;
                while (left < right)
                {
                    int mid = (left + right + 1) / 2;
                    if (GetSum(index, mid, n) <= maxSum)
                    {
                        left = mid;
                    }
                    else
                    {
                        right = mid - 1;
                    }
                }

                return left;
            }
            private long GetSum(int index, int value, int n)
            {
                long count = 0;

                // On index's left:
                // If value > index, there are index + 1 numbers in the arithmetic sequence:
                // [value - index, ..., value - 1, value].
                // Otherwise, there are value numbers in the arithmetic sequence:
                // [1, 2, ..., value - 1, value], plus a sequence of length (index - value + 1) of 1s. 
                if (value > index)
                {
                    count += (long)(value + value - index) * (index + 1) / 2;
                }
                else
                {
                    count += (long)(value + 1) * value / 2 + index - value + 1;
                };
                // On index's right:
                // If value >= n - index, there are n - index numbers in the arithmetic sequence:
                // [value, value - 1, ..., value - n + 1 + index].
                // Otherwise, there are value numbers in the arithmetic sequence:
                // [value, value - 1, ..., 1], plus a sequence of length (n - index - value) of 1s. 
                if (value >= n - index)
                {
                    count += (long)(value + value - n + 1 + index) * (n - index) / 2;
                }
                else
                {
                    count += (long)(value + 1) * value / 2 + n - index - value;
                }

                return count - value;

            }

        }

        /* 1885. Count Pairs in Two Arrays
        https://leetcode.com/problems/count-pairs-in-two-arrays/description/
         */
        class CountPairsInTwoArraysSol
        {
            /* Approach 1: Binary Search
Complexity Analysis
Let n be the length of nums1 and nums2.
•	Time complexity: O(nlogn)
Calculating the difference of each pair of elements nums1[i] and nums2[i] takes O(n).
difference is of size n, so sorting difference takes O(nlogn).
Counting the number of valid pairs using binary search takes O(nlogn). The outer loop runs n times, once for each element in difference, and the inner loop can run up to logn times since we divide the search space in half with each iteration.
The total time complexity will be O(n+2nlogn), which we can simplify to O(nlogn).
•	Space complexity: O(n)
We use a few variables and the array difference, which is size O(n)
Note that some extra space is used when we sort an array in place. The space complexity of the sorting algorithm depends on the programming language.
o	In Python, the sort method sorts a list using the Tim Sort algorithm which is a combination of Merge Sort and Insertion Sort and has O(n) additional space. Additionally, Tim Sort is designed to be a stable algorithm.
o	In Java, Arrays.sort() is implemented using a variant of the Quick Sort algorithm which has a space complexity of O(logn) for sorting an array.
o	In C++, the sort() function is implemented as a hybrid of Quick Sort, Heap Sort, and Insertion Sort, with a worse-case space complexity of O(logn).
The dominating term is O(n).

             */
            public long UsingBinarySearch(int[] nums1, int[] nums2)
            {
                int N = nums1.Length;  // nums2 is the same length

                // Difference[i] stores nums1[i] - nums2[i]
                long[] difference = new long[N];
                for (int i = 0; i < N; i++)
                {
                    difference[i] = nums1[i] - nums2[i];
                }
                Array.Sort(difference);

                // Count the number of valid pairs
                long result = 0;
                for (int i = 0; i < N; i++)
                {
                    // All indices j following i make a valid pair
                    if (difference[i] > 0)
                    {
                        result += N - i - 1;
                    }
                    else
                    {
                        // Binary search to find the first index j
                        // that makes a valid pair with i
                        int left = i + 1;
                        int right = N - 1;
                        while (left <= right)
                        {
                            int mid = left + (right - left) / 2;
                            // If difference[mid] is a valid pair, search in left half
                            if (difference[i] + difference[mid] > 0)
                            {
                                right = mid - 1;
                                // If difference[mid] does not make a valid pair, search in right half
                            }
                            else
                            {
                                left = mid + 1;
                            }
                        }
                        // After the search left points to the first index j that makes
                        // a valid pair with i so we count that and all following indices
                        result += N - left;
                    }
                }
                return result;
            }

            /* Approach 2: Sort and Two Pointer 
            Complexity Analysis
Let n be the length of nums1 and nums2.
•	Time complexity: O(nlogn)
Calculating the difference of each pair of elements nums1[i] and nums2[i] takes O(n).
difference is of size n, so sorting difference takes O(nlogn).
Counting the number of valid pairs takes O(n), since we use two pointers to traverse the array, one from the beginning and one from the end. One of the pointers is moved one step to the center with each iteration. Iteration stops when they meet in the middle.
The total time complexity will be O(2n+nlogn), which we can simplify to O(nlogn).
•	Space complexity: O(n)
We use a few variables and the array difference, which is size O(n)
Note that some extra space is used when we sort an array in place. The space complexity of the sorting algorithm depends on the programming language.
o	In Python, the sort method sorts a list using the Tim Sort algorithm which is a combination of Merge Sort and Insertion Sort and has O(n) additional space. Additionally, Tim Sort is designed to be a stable algorithm.
o	In Java, Arrays.sort() is implemented using a variant of the Quick Sort algorithm which has a space complexity of O(logn) for sorting an array.
o	In C++, the sort() function is implemented as a hybrid of Quick Sort, Heap Sort, and Insertion Sort, with a worse-case space complexity of O(logn).
The dominating term is O(n).

            */
            public long UsingSortingAndTwoPointers(int[] nums1, int[] nums2)
            {
                int N = nums1.Length; // nums2 is the same length

                // Difference[i] stores nums1[i] - nums2[i]
                long[] difference = new long[N];
                for (int i = 0; i < N; i++)
                {
                    difference[i] = nums1[i] - nums2[i];
                }
                Array.Sort(difference);

                // Count the number of valid pairs
                long result = 0;
                int left = 0;
                int right = N - 1;
                while (left < right)
                {
                    // Left makes a valid pair with right
                    // Right also makes a valid pair with the indices between the pointers
                    if (difference[left] + difference[right] > 0)
                    {
                        result += right - left;
                        right--;
                        // Left and right are not a valid pair
                    }
                    else
                    {
                        left++;
                    }
                }
                return result;
            }
        }

        /* 845. Longest Mountain in Array
        https://leetcode.com/problems/longest-mountain-in-array/description/
         */
        public class LongestMountainInArraySol
        {

            /* Approach #1: Two Pointer [Accepted]
Complexity Analysis
•	Time Complexity: O(N), where N is the length of A.
•	Space Complexity: O(1).
•	Can you solve it in O(1) space?

             */
            public int LongestMountain(int[] array)
            {
                int arrayLength = array.Length;
                int longestMountainLength = 0, baseIndex = 0;
                while (baseIndex < arrayLength)
                {
                    int endIndex = baseIndex;
                    // if baseIndex is a left-boundary
                    if (endIndex + 1 < arrayLength && array[endIndex] < array[endIndex + 1])
                    {
                        // set endIndex to the peak of this potential mountain
                        while (endIndex + 1 < arrayLength && array[endIndex] < array[endIndex + 1]) endIndex++;

                        // if endIndex is really a peak..
                        if (endIndex + 1 < arrayLength && array[endIndex] > array[endIndex + 1])
                        {
                            // set endIndex to the right-boundary of mountain
                            while (endIndex + 1 < arrayLength && array[endIndex] > array[endIndex + 1]) endIndex++;
                            // record candidate answer
                            longestMountainLength = Math.Max(longestMountainLength, endIndex - baseIndex + 1);
                        }
                    }

                    baseIndex = Math.Max(endIndex, baseIndex + 1);
                }

                return longestMountainLength;
            }
        }






    }


}


