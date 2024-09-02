using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.Design
{
    /*
    155. Min Stack
    https://leetcode.com/problems/min-stack/	

    */
    public class MinStack
    {
        /*
        Approach 1: Stack of Value/ Minimum Pairs

        Let n be the total number of operations performed.
        Time Complexity : O(1) for all operations.
            push(...): Checking the top of a Stack, comparing numbers, and pushing to the top of a Stack (or adding to the end of an Array or List) are all O(1) operations. Therefore, this overall is an O(1) operation.
            pop(...): Popping from a Stack (or removing from the end of an Array, or List) is an O(1) operation.
            top(...): Looking at the top of a Stack is an O(1) operation.
            getMin(...): Same as above. This operation is O(1) because we do not need to compare values to find it. If we had not kept track of it on the Stack, and instead had to search for it each time, the overall time complexity would have been O(n).
        Space Complexity : O(n).Worst case is that all the operations are push. In this case, there will be O(2⋅n)=O(n) space used.


        */
        public class MinStackPairs
        {
            private Stack<int[]> stack = new Stack<int[]>();

            public MinStackPairs()
            {
            }

            public void Push(int x)
            {
                // If the Stack is empty then min value is the first value we add
                if (stack.Count == 0)
                {
                    stack.Push(new int[] { x, x });
                    return;
                }

                int current_min = stack.Peek()[1];
                stack.Push(new int[] { x, Math.Min(x, current_min) });
            }

            public void Pop()
            {
                stack.Pop();
            }

            public int Top()
            {
                return stack.Peek()[0];
            }

            public int GetMin()
            {
                return stack.Peek()[1];
            }
        }
        /*
        Approach 2: Two Stacks

        Let n be the total number of operations performed.
        Time Complexity : O(1) for all operations.
            push(...): Checking the top of a Stack, comparing numbers, and pushing to the top of a Stack (or adding to the end of an Array or List) are all O(1) operations. Therefore, this overall is an O(1) operation.
            pop(...): Popping from a Stack (or removing from the end of an Array, or List) is an O(1) operation.
            top(...): Looking at the top of a Stack is an O(1) operation.
            getMin(...): Same as above. This operation is O(1) because we do not need to compare values to find it. If we had not kept track of it on the Stack, and instead had to search for it each time, the overall time complexity would have been O(n).
        Space Complexity : O(n).Worst case is that all the operations are push. In this case, there will be O(2⋅n)=O(n) space used.
        */
        public class MinStackTwoStacks
        {
            Stack<int> stack;
            Stack<int> minStack;

            public MinStackTwoStacks()
            {
                stack = new Stack<int>();
                minStack = new Stack<int>();
            }

            public void Push(int x)
            {
                stack.Push(x);
                if (minStack.Count == 0 || x <= minStack.Peek()) minStack.Push(x);
            }

            public void Pop()
            {
                if (stack.Peek() == minStack.Peek()) minStack.Pop();
                stack.Pop();
            }

            public int Top()
            {
                return stack.Peek();
            }

            public int GetMin()
            {
                return minStack.Peek();
            }
        }
        /*
        Approach 3: Improved Two Stacks

        Let n be the total number of operations performed.
        Time Complexity : O(1) for all operations.
            push(...): Checking the top of a Stack, comparing numbers, and pushing to the top of a Stack (or adding to the end of an Array or List) are all O(1) operations. Therefore, this overall is an O(1) operation.
            pop(...): Popping from a Stack (or removing from the end of an Array, or List) is an O(1) operation.
            top(...): Looking at the top of a Stack is an O(1) operation.
            getMin(...): Same as above. This operation is O(1) because we do not need to compare values to find it. If we had not kept track of it on the Stack, and instead had to search for it each time, the overall time complexity would have been O(n).
        Space Complexity : O(n).Worst case is that all the operations are push. In this case, there will be O(2⋅n)=O(n) space used.
        */
        public class MinStackTwoStackOptimal
        {
            private Stack<int> stack = new Stack<int>();
            private Stack<int[]> minStack = new Stack<int[]>();

            public MinStackTwoStackOptimal()
            {
            }

            public void Push(int x)
            {
                // We always put the number onto the main stack.
                stack.Push(x);

                // If the min stack is empty, or this number is smaller than
                // the top of the min stack, put it on with a count of 1.
                if (minStack.Count == 0 || x < minStack.Peek()[0])
                {
                    minStack.Push(new int[] { x, 1 });
                }
                // Else if this number is equal to what's currently at the top
                // of the min stack, then increment the count at the top by 1.
                else if (x == minStack.Peek()[0])
                {
                    minStack.Peek()[1]++;
                }
            }

            public void Pop()
            {
                // If the top of min stack is the same as the top of stack
                // then we need to decrement the count at the top by 1.
                if (stack.Peek() == minStack.Peek()[0])
                {
                    minStack.Peek()[1]--;
                }

                // If the count at the top of min stack is now 0, then remove
                // that value as we're done with it.
                if (minStack.Peek()[1] == 0)
                {
                    minStack.Pop();
                }

                // And like before, pop the top of the main stack.
                stack.Pop();
            }

            public int Top()
            {
                return stack.Peek();
            }

            public int GetMin()
            {
                return minStack.Peek()[0];
            }
        }

    }
}