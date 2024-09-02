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
            //2.Optimal - T: O(n) | O(n)
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
        public float MedianOfTwoSortedArrays(int[] arrayOne, int[] arrayTwo)
        {
        TODO:

            //1. Naive - T:O(n+m) | O(1)
            float median = MedianOfTwoSortedArraysNaive(arrayOne, arrayTwo);

            //2Optimal.T:O(log(min(n+m))) | O(1) - using Binary Search variation
            median = MedianOfTwoSortedArraysOptimal(arrayOne, arrayTwo);

            return median;

        }

        private float MedianOfTwoSortedArraysOptimal(int[] arrayOne, int[] arrayTwo)
        {
            int[] smallArray = arrayOne.Length <= arrayTwo.Length ? arrayOne : arrayTwo;
            int[] bigArray = arrayOne.Length >= arrayTwo.Length ? arrayOne : arrayTwo;

            int leftIdx = 0;
            int rightIdx = smallArray.Length - 1;
            int mergeLeftIdx = (smallArray.Length + bigArray.Length - 1) / 2;
            while (true)
            {

                int smallPartitionIdx = (int)Math.Floor((double)(leftIdx + rightIdx) / 2);

                int bigPartitionIdx = mergeLeftIdx - smallPartitionIdx - 1;

                int smallMaxLeftValue = smallPartitionIdx >= 0 ? smallArray[smallPartitionIdx] : Int32.MinValue;
                int smallMinRightValue = smallPartitionIdx + 1 < smallArray.Length ? smallArray[smallPartitionIdx + 1] : Int32.MaxValue;


                int bigMaxLeftValue = bigPartitionIdx >= 0 ? bigArray[bigPartitionIdx] : Int32.MinValue;
                int bigMinRightValue = bigPartitionIdx + 1 < bigArray.Length ? bigArray[bigPartitionIdx + 1] : Int32.MaxValue;

                if (smallMaxLeftValue > bigMinRightValue)
                {
                    rightIdx = smallPartitionIdx - 1;
                }
                else if (bigMaxLeftValue > smallMinRightValue)
                {
                    leftIdx = smallPartitionIdx + 1;
                }
                else
                {
                    if ((smallArray.Length + bigArray.Length) % 2 == 0)
                    {
                        return (float)(Math.Max(smallMaxLeftValue, bigMaxLeftValue) +
                                        Math.Min(smallMinRightValue, bigMinRightValue)) / 2;
                    }
                    return Math.Max(smallMaxLeftValue, bigMaxLeftValue);
                }
            }

        }

        private float MedianOfTwoSortedArraysNaive(int[] arrayOne, int[] arrayTwo)
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



    }
}