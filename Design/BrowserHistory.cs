using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.Design
{
    /*
    1472. Design Browser History
    https://leetcode.com/problems/design-browser-history/

    */
    public class BrowserHistory
    {
        /*
        Approach 1: Two Stacks
        Complexity Analysis
        Let's assume here, n visit calls are made, m is the maximum number of steps to go forward or back, and l is the maximum length of the URL string.
        •	Time complexity:
                o	In the visit(url) method, we push the URL string in the history stack, assign the given url string as the current URL, and then we clear the future stack, all these operations take O(1) time each.
                Thus, in the worst case each call to the visit(url) method will take O(1) time.
                o	In the back(steps) and forward(steps) methods, we push and pop strings in the future and history stacks. We do these two operations unless we are done with m steps or all elements are removed from the stack which might have n elements in it.
                Thus, in the worst case, each call to these methods will take O(min(m,n)) time.
                Note: In C++, the stack is implemented using vectors, the push operation is simply updating the stack pointer and copying the string. The underlying vector class takes care of the reallocation and copy of the string, so the push operation is still an O(1) operation (we will discuss this in detail in the last approach).
                Similarly, in the case of Java, Python and most languages, the push operation on stack implemented using a dynamic array or list is O(1) as it is only updating the stack pointer and not copying the string.
        •	Space complexity:
                o	We might visit n URL strings and they will be stored in our stacks.
                o	Thus, in the worse case, we use O(l⋅n) space.

        */
        class BrowserHistoryTwoStacks
        {
            private Stack<string> history;
            private Stack<string> future;
            private string current;

            public BrowserHistoryTwoStacks(string homepage)
            {
                history = new Stack<string>();
                future = new Stack<string>();
                // 'homepage' is the first visited URL.
                current = homepage;
            }

            public void Visit(string url)
            {
                // Push 'current' in 'history' stack and mark 'url' as 'current'.
                history.Push(current);
                current = url;
                // We need to delete all entries from 'future' stack.
                future.Clear();
            }

            public string Back(int steps)
            {
                // Pop elements from 'history' stack, and push elements in 'future' stack.
                while (steps > 0 && history.Count > 0)
                {
                    future.Push(current);
                    current = history.Pop();
                    steps--;
                }
                return current;
            }

            public string Forward(int steps)
            {
                // Pop elements from 'future' stack, and push elements in 'history' stack.
                while (steps > 0 && future.Count > 0)
                {
                    history.Push(current);
                    current = future.Pop();
                    steps--;
                }
                return current;
            }
        }
        /*
            Approach 2: Doubly Linked List (DLL)
            Complexity Analysis
            Let's assume here, n visit calls are made, m is the maximum number of steps to go forward or back, and l is the maximum length of the URL string.
            •	Time complexity:
                o	In the visit(url) method, we insert a new node in our doubly linked list, it will take O(l) time to create a new node (to allocate memory for l characters of the url string), and then we mark this new node as current which will take O(1) time.
                Thus, in the worst case each call to the visit(url) method will take O(l) time.
                o	In the back(steps) and forward(steps) methods, we iterate on our doubly linked list nodes and stop when m nodes are iterated or we reached the end.
                Thus, in the worst case, each call to these methods will take O(min(m,n)) time.
            •	Space complexity:
                o	We might visit n URL strings and they will be stored in our doubly linked list.
                o	Thus, in the worse case, we use O(l⋅n) space.
            */
        public class DLLNode
        {
            public string Data;
            public DLLNode Previous;
            public DLLNode Next;

            public DLLNode(string url)
            {
                Data = url;
                Previous = null;
                Next = null;
            }
        }

        public class BrowserHistoryDLL
        {
            private DLLNode linkedListHead;
            private DLLNode current;

            public BrowserHistoryDLL(string homepage)
            {
                // 'homepage' is the first visited URL.
                linkedListHead = new DLLNode(homepage);
                current = linkedListHead;
            }

            public void Visit(string url)
            {
                // Insert new node 'url' in the right of current node.
                DLLNode newNode = new DLLNode(url);
                current.Next = newNode;
                newNode.Previous = current;
                // Make this new node as current node now.
                current = newNode;
            }

            public string Back(int steps)
            {
                // Move 'current' pointer in left direction.
                while (steps > 0 && current.Previous != null)
                {
                    current = current.Previous;
                    steps--;
                }
                return current.Data;
            }

            public string Forward(int steps)
            {
                // Move 'current' pointer in right direction.
                while (steps > 0 && current.Next != null)
                {
                    current = current.Next;
                    steps--;
                }
                return current.Data;
            }
        }

        /*
            Approach 3: Dynamic Array(DA)
            Complexity Analysis
            Let's assume here, n visit calls are made, m is the maximum number of steps to go forward or back, and l is the maximum length of a URL string.
            •	Time complexity:
                o	In the visit(url) method, we insert the URL string in our array and update the current pointer, both of these operations will take O(1) time each.
                Thus, in the worst case each call to the visit(url) method will take O(1) time.
                o	In the back(steps) and forward(steps) methods, we directly return the element at the required index which takes O(1) time.
                Thus, in the worst case, each call to these methods will take O(1) time.
                Note: The time complexity of the push operation as a whole is not just the time taken for storing the string. It also includes the time complexity of resizing the vector if it is full, and adjusting the internal pointers and indices.
                Due to internal optimizations, the pushing of strings, integers, etc. takes very little time (it reuses their existing memory blocks internally instead of creating a new block for each push operation) compared to the time taken for reallocation of memory.
                If the vector is full, it needs to be resized to make room for the new elements. This typically involves allocating a new block of memory and copying the existing elements from the old block to the new block, thus the average time complexity when doing a lot of push operations does not depends on the length of the string but depends on time taken for reallocation.
                The time complexity of reallocation is dependent on the size of the vector, but as it happens rarely thus it is usually amortized over multiple push operations, meaning that the average time complexity of a push operation is still O(1).

                In summary, the time complexity of pushing a string of length l into a vector is usually amortized over multiple push operations, so the overall time complexity of each push operation is O(1).
                This applies to C++, Java, Python, and the majority of other languages. You can read more about this amortized push/append operation behavior here.
            •	Space complexity:
                o	We might visit n URL strings and they will be stored in our array.
                o	Thus, in the worse case, we use O(l⋅n) space.
        */
        class BrowserHistoryDA
        {
            List<string> visitedURLs;
            int currentURLIndex, lastURLIndex;

            public BrowserHistoryDA(string homepage)
            {
                // 'homepage' is the first visited URL.
                visitedURLs = new List<string> { homepage };
                currentURLIndex = 0;
                lastURLIndex = 0;
            }

            public void Visit(string url)
            {
                currentURLIndex += 1;
                if (visitedURLs.Count > currentURLIndex)
                {
                    // We have enough space in our array to overwrite an old 'url' entry with new one.
                    visitedURLs[currentURLIndex] = url;
                }
                else
                {
                    // We have to insert a new 'url' entry at the end.
                    visitedURLs.Add(url);
                }
                // This 'url' will be last URL if we try to go forward.
                lastURLIndex = currentURLIndex;
            }

            public string Back(int steps)
            {
                // Move 'currentURLIndex' pointer in left direction.
                currentURLIndex = Math.Max(0, currentURLIndex - steps);
                return visitedURLs[currentURLIndex];
            }

            public string Forward(int steps)
            {
                // Move 'currentURLIndex' pointer in right direction.
                currentURLIndex = Math.Min(lastURLIndex, currentURLIndex + steps);
                return visitedURLs[currentURLIndex];
            }
        }


    }
    /**
 * Your BrowserHistory object will be instantiated and called as such:
 * BrowserHistory obj = new BrowserHistory(homepage);
 * obj.Visit(url);
 * string param_2 = obj.Back(steps);
 * string param_3 = obj.Forward(steps);
 */
}