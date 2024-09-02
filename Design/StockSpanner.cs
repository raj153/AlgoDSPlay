using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.Design
{
    /*
    901. Online Stock Span
    https://leetcode.com/problems/online-stock-span/

    Complexity Analysis
    Given n as the number of calls to next,
    •	Time complexity of each call to next: O(1)
            Even though there is a while loop in next, that while loop can only run n times total across the entire algorithm. Each element can only be popped off the stack once, and there are up to n elements.
            This is called amortized analysis - if you average out the time it takes for next to run across n calls, it works out to be O(1). If one call to next takes a long time because the while loop runs many times, then the other calls to next won't take as long because their while loops can't run as long.
    •	Space complexity: O(n)
            In the worst case scenario for space (when all the stock prices are decreasing), the while loop will never run, which means the stack grows to a size of n.

    */
    public class StockSpanner {
    Stack<int[]> stack = new Stack<int[]>();
    
    public int next(int price) {
        int ans = 1;
        while (stack.Count >0 && stack.Peek()[0] <= price) {
            ans += stack.Pop()[1];
        
        }
        stack.Push(new int[] {price, ans});
        return ans;
    }
}

/**
 * Your StockSpanner object will be instantiated and called as such:
 * StockSpanner obj = new StockSpanner();
 * int param_1 = obj.next(price);
 */
}