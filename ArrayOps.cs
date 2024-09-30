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
using AlgoDSPlay.DataStructures;

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
Here, N is the total number of strings in the array words, namely words.length, and M is the length of the longest string in the array words.
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
Here, N is the total number of strings in the array words, namely words.length, and M is the length of the longest string in the array words.
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































    }


}


