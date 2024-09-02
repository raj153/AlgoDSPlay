using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.Design
{
    /*
    716. Max Stack
    https://leetcode.com/problems/max-stack/description/
    */
    public class MaxStack
    {
        /*
        Approach 1: Two Balanced Trees

        Let N be the number of elements to add to the stack.
        Time Complexity: O(logN) for each operation except for initialization. All operations other than initialization are involved with finding/inserting/removing elements in a balanced tree once or twice. 
                         In general, the upper bound of time complexity for each of them is O(logN). However, note that top and peekMax operations, requiring only the last element in a balanced tree, 
                         can be even done in O(1) with set::rbegin() in C++ and some special handles on the last element of SortedList in Python. 
                         But last for TreeSet in Java haven't implemented similar optimization yet, we have to get the last element in O(logN).
        Space Complexity: O(N), the maximum size of the two balanced trees.
        */
        class MaxStackTwoBT
        {

            private SortedSet<int[]> stack;
            private SortedSet<int[]> values;
            private int cnt;

            public MaxStackTwoBT()
            {
                Comparer<int[]> comp = Comparer<int[]>.Create((a, b)=> {
                    return a[0] == b[0] ? a[1] - b[1] : a[0] - b[0];
                });
                stack = new SortedSet<int[]>(comp);
                values = new SortedSet<int[]>(comp);
                cnt = 0;
            }

            public void Push(int x)
            {
                stack.Add(new int[] { cnt, x });
                values.Add(new int[] { x, cnt });
                cnt++;
            }

            public int Pop()
            {
                int[] pair = stack.Last();
                values.Remove(new int[] { pair[1], pair[0] });
                return pair[1];
            }

            public int Top()
            {
                return stack.Last()[1];
            }

            public int PeekMax()
            {
                return values.Last()[0];
            }

            public int PopMax()
            {
                int[] pair = values.Last();
                stack.Remove(new int[] { pair[1], pair[0] });
                return pair[0];
            }
        }

        /*
        Approach 2: Heap + Lazy Update

        Let N be the number of elements to add to the stack.
        Time Complexity:
            push: O(logN), it costs O(logN) to add an element to heap and O(1) to add an it to stack.
                  The amortized time complexity of operations caused by a single pop/popMax call is O(logN). For a pop call, we first remove the last element in stack and add its ID to removed in O(1), and result in a deletion of the top element in heap in the future (when peekMax or popMax is called), which has a time complexity of logN. Similarly, popMax needs O(logN) immediately and O(1) in the operations later. Note that because we lazy-update the two data structures, future operations might never happen in some cases. But even in the worst cases, the upper bound of the amortized time complexity is still only O(logN).
            top: O(1), excluding the time cost related to popMax calls we discussed above.
            peekMax: O(logN), excluding the time cost related to pop calls we discussed above.
        Space Complexity: O(N), the maximum size of the heap, stack, and removed.

        */
        class MaxStackHeap
        {
            //TODO : Fix Comparer below

           /*  private Stack<int[]> stack;
            private PriorityQueue<int[], int[]> heap;
            private HashSet<int> removed;
            private int cnt;

            public MaxStackHeap()
            {
                IComparer<int[]> comp = Comparer<int[]]>.Create((a, b)=> {
                    return a[0] == b[0] ? a[1] - b[1] : a[0] - b[0];
                });
                stack = new Stack<int[]>();
                heap = new PriorityQueue<int[], int>(comp);
                cnt = 0;
            }

            public void Push(int x)
            {
                stack.Add(new int[] { cnt, x });
                values.Add(new int[] { x, cnt });
                cnt++;
            }

            public int Pop()
            {
                int[] pair = stack.Last();
                values.Remove(new int[] { pair[1], pair[0] });
                return pair[1];
            }

            public int Top()
            {
                return stack.Last()[1];
            }

            public int PeekMax()
            {
                return values.Last()[0];
            }

            public int PopMax()
            {
                int[] pair = values.Last();
                stack.Remove(new int[] { pair[1], pair[0] });
                return pair[0];
            }
 */        }
    }
    }

    
 