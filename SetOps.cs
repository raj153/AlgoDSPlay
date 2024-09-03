using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AlgoDSPlay
{
    public class SetOps
    {
        //https://www.algoexpert.io/questions/powerset
        public static List<List<int>> GeneratePowerSet(List<int> array)
        {


            //1.Iterative - T:O(n*2^n) | O(n*2^n)
            List<List<int>> powersets = GeneratePowerSetIterative(array);

            //2.Recursive - T:O(n*2^n) | O(n*2^n)
            powersets = GeneratePowerSetRecursive(array, array.Count - 1);
            return powersets;
        }

        private static List<List<int>> GeneratePowerSetRecursive(List<int> array, int idx)
        {
            if (idx < 0)
            {
                List<List<int>> emptySet = new List<List<int>>();
                emptySet.Add(new List<int>());
                return emptySet;
            }

            int element = array[idx];
            List<List<int>> subsets = GeneratePowerSetRecursive(array, idx - 1);
            int length = subsets.Count;
            for (int i = 0; i < length; i++)
            {
                List<int> currentSubset = new List<int>(subsets[i]);
                currentSubset.Add(element);
                subsets.Add(currentSubset);
            }
            return subsets;
        }

        private static List<List<int>> GeneratePowerSetIterative(List<int> array)
        {

            List<List<int>> subSets = new List<List<int>>();
            subSets.Add(new List<int>());

            foreach (var element in array)
            {
                int length = subSets.Count;

                for (int i = 0; i < length; i++)
                {
                    List<int> currentSubset = new List<int>(subSets[i]);
                    currentSubset.Add(element);
                    subSets.Add(currentSubset);

                }
            }
            return subSets;

        }
        /*
        78. Subsets
https://leetcode.com/problems/subsets/description/

        */
        public List<List<int>> Subsets(int[] nums)
        {

            /*
            Approach 1: Cascading(Iterative)
Complexity Analysis
•	Time complexity: O(N×2^N) to generate all subsets and then copy them into the output list.
•	Space complexity: O(N×2^N). This is exactly the number of solutions for subsets multiplied by the number N of elements to keep for each subset.
o	For a given number, it could be present or absent (i.e. binary choice) in a subset solution. As a result, for N numbers, we would have in total 2N choices (solutions).

            
            */
            var subsets = SubsetsCascading(nums);

            /*
Approach 2: Backtracking
Complexity Analysis
•	Time complexity: O(N×2^N) to generate all subsets and then copy them into the output list.
•	Space complexity: O(N). We are using O(N) space to maintain curr, and are modifying curr in-place with backtracking. Note that for space complexity analysis, we do not count space that is only used for the purpose of returning output, so the output array is ignored.

            */
            subsets = SubsetsBacktrack(nums);

            return subsets;


        }
        public List<List<int>> SubsetsCascading(int[] nums)
        {
            List<List<int>> output = new List<List<int>>();
            output.Add(new List<int>());

            foreach (int num in nums)
            {
                List<List<int>> newSubsets = new List<List<int>>();
                foreach (List<int> curr in output)
                {
                    List<int> temp = new List<int>(curr);
                    temp.Add(num);
                    newSubsets.Add(temp);
                }
                foreach (List<int> curr in newSubsets)
                {
                    output.Add(curr);
                }
            }
            return output;
        }

        private List<List<int>> output = new List<List<int>>();
        private int n, k;

        private void Backtrack(int first, List<int> curr, int[] nums)
        {
            if (curr.Count == k)
                output.Add(new List<int>(curr));
            for (int i = first; i < n; ++i)
            {
                curr.Add(nums[i]);
                Backtrack(i + 1, curr, nums);
                curr.RemoveAt(curr.Count - 1);
            }
        }

        public List<List<int>> SubsetsBacktrack(int[] nums)
        {
            n = nums.Length;
            for (k = 0; k < n + 1; ++k)
            {
                List<int> currCombo = new List<int>();
                Backtrack(0, currCombo, nums);
            }
            return output;
        }

        /*
        90. Subsets II
        https://leetcode.com/problems/subsets-ii/description/

        */
        public IList<IList<int>> SubsetsWithDup(int[] nums)
        {
            /*
Approach 1: Bitmasking
Complexity Analysis
Here n is the size of the nums array.
•	Time complexity: O(n⋅2^n)
Sorting the nums array requires nlogn time. The outer for loop runs 2n times. The inner loop runs n times. We know that average case time complexity for set operations is O(1). Although, to generate a hash value for each subset O(n) time will be required. However, we are generating the hashcode while iterating the inner for loop. So the overall time complexity is O(nlogn+2^n⋅(n (for inner loop) + n (for stringbuilder to string conversion in Java) )) = O(2n⋅^n).
•	Space complexity: O(n⋅2^n)
We need to store at most 2^n number of subsets in the set, seen. The maximum length of any subset can be n.
The space complexity of the sorting algorithm depends on the implementation of each programming language. For instance, in Java, the Arrays.sort() for primitives is implemented as a variant of quicksort algorithm whose space complexity is O(logn). In C++ sort() function provided by STL is a hybrid of Quick Sort, Heap Sort and Insertion Sort with the worst case space complexity of O(logn). Thus the use of inbuilt sort() function adds O(logn) to space complexity.
The space required for the output list is not considered while analyzing space complexity. Thus the overall space complexity in Big O Notation is O(n⋅2n).
           
            */
            IList<IList<int>> subSets = SubsetsWithDupBitMasking(nums);

            /*
Approach 2: Cascading (Iterative)
Complexity Analysis
Here n is the number of elements present in the given array.
•	Time complexity: O(n⋅2^n)
At first, we need to sort the given array which adds O(nlogn) to the time complexity. Next, we use two for loops to create all possible subsets. In the worst case, i.e., with an array of n distinct integers, we will have a total of 2^n subsets. Thus the two for loops will add O(2n) to time complexity. Also in the inner loop, we deep copy the previously generated subset before adding the current integer (to create a new subset). This in turn requires the time of order n as the maximum number of elements in the currentSubset will be at most n. Thus, the time complexity in the subset generation step using two loops is O(n⋅2^n). Thereby, the overall time complexity is O(nlogn+n⋅2^n) = O(n⋅(logn+2^n)) ~ O(n⋅2^n).
•	Space complexity: O(logn)
The space complexity of the sorting algorithm depends on the implementation of each programming language. For instance, in Java, the Arrays.sort() for primitives is implemented as a variant of quicksort algorithm whose space complexity is O(logn). In C++ sort() function provided by STL is a hybrid of Quick Sort, Heap Sort and Insertion Sort with the worst case space complexity of O(logn). Thus the use of inbuilt sort() function adds O(logn) to space complexity.
The space required for the output list is not considered while analyzing space complexity. Thus the overall space complexity in Big O Notation is O(logn).
            
            */
            subSets = SubsetsWithDupCascading(nums);
            /*
Approach 3: Backtracking
Complexity Analysis
Here n is the number of elements present in the given array.
•	Time complexity: O(n⋅2^n)
As we can see in the diagram above, this approach does not generate any duplicate subsets. Thus, in the worst case (array consists of n distinct elements), the total number of recursive function calls will be 2^n. Also, at each function call, a deep copy of the subset currentSubset generated so far is created and added to the subsets list. This will incur an additional O(n) time (as the maximum number of elements in the currentSubset will be n). So overall, the time complexity of this approach will be O(n⋅2^n).
•	Space complexity: O(n)
The space complexity of the sorting algorithm depends on the implementation of each programming language. For instance, in Java, the Arrays.sort() for primitives is implemented as a variant of quicksort algorithm whose space complexity is O(logn). In C++ sort() function provided by STL is a hybrid of Quick Sort, Heap Sort and Insertion Sort with the worst case space complexity of O(logn). Thus the use of inbuilt sort() function adds O(logn) to space complexity.
The recursion call stack occupies at most O(n) space. The output list of subsets is not considered while analyzing space complexity. So, the space complexity of this approach is O(n).

            */
            subSets = SubsetsWithDupBacktrack(nums);

        return subSets;

        }

        public IList<IList<int>> SubsetsWithDupBitMasking(int[] nums)
        {
            IList<IList<int>> subsets = new List<IList<int>>();
            int n = nums.Length;
            // Sort the generated subset. This will help to identify duplicates.
            Array.Sort(nums);
            int maxNumberOfSubsets = (int)Math.Pow(2, n);
            // To store the previously seen sets.
            HashSet<string> seen = new HashSet<string>();
            for (int subsetIndex = 0; subsetIndex < maxNumberOfSubsets;
                 subsetIndex++)
            {
                // Append subset corresponding to that bitmask.
                List<int> currentSubset = new List<int>();
                StringBuilder hashcode = new StringBuilder();
                for (int j = 0; j < n; j++)
                {
                    // Generate the bitmask
                    int mask = 1 << j;
                    int isSet = mask & subsetIndex;
                    if (isSet != 0)
                    {
                        currentSubset.Add(nums[j]);
                        // Generate the hashcode by creating a comma separated
                        // string of numbers in the currentSubset.
                        hashcode.Append(nums[j]).Append(",");
                    }
                }

                if (!seen.Contains(hashcode.ToString()))
                {
                    seen.Add(hashcode.ToString());
                    subsets.Add(currentSubset);
                }
            }

            return subsets;
        }
        public IList<IList<int>> SubsetsWithDupCascading(int[] nums)
        {
            Array.Sort(nums);
            List<IList<int>> subsets = new List<IList<int>>();
            subsets.Add(new List<int>());
            int subsetSize = 0;
            for (int i = 0; i < nums.Length; i++)
            {
                // subsetSize refers to the size of the subset in the previous step.
                // This value also indicates the starting index of the subsets
                // generated in this step.
                int startingIndex =
                    (i >= 1 && nums[i] == nums[i - 1]) ? subsetSize : 0;
                subsetSize = subsets.Count;
                for (int j = startingIndex; j < subsetSize; j++)
                {
                    List<int> currentSubset = new List<int>(subsets[j]);
                    currentSubset.Add(nums[i]);
                    subsets.Add(currentSubset);
                }
            }

            return subsets;
        }
        public IList<IList<int>> SubsetsWithDupBacktrack(int[] nums)
        {
            Array.Sort(nums);
            IList<IList<int>> subsets = new List<IList<int>>();
            IList<int> currentSubset = new List<int>();
            SubsetsWithDupHelper(subsets, currentSubset, nums, 0);
            return subsets;
        }

        private void SubsetsWithDupHelper(IList<IList<int>> subsets,
                                          IList<int> currentSubset, int[] nums,
                                          int index)
        {
            // Add the subset formed so far to the subsets list.
            subsets.Add(new List<int>(currentSubset));
            for (int i = index; i < nums.Length; i++)
            {
                // If the current element is a duplicate, ignore.
                if (i != index && nums[i] == nums[i - 1])
                {
                    continue;
                }

                currentSubset.Add(nums[i]);
                SubsetsWithDupHelper(subsets, currentSubset, nums, i + 1);
                currentSubset.RemoveAt(currentSubset.Count - 1);
            }
        }


    }
}