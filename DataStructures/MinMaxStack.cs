using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.DataStructures
{
    //https://www.algoexpert.io/questions/min-max-stack-construction
    public class MinMaxStack
    {
        List<Dictionary<string, int>> minMaxStack = new List<Dictionary<string, int>>();

        List<int> stack = new List<int>();

        //T:O(1) | S:O(1)
        public int Peek(){
            return stack[stack.Count-1];
        }

        //T:O(1) | S:O(1)
        public int Pop(){
            int val = stack[stack.Count-1];
            stack.RemoveAt(stack.Count-1);
            minMaxStack.RemoveAt(minMaxStack.Count-1);
            return val;
        }

        //T:O(1) | S:O(1)
        public void Push(int val){
            Dictionary<string, int> newMinMax = new Dictionary<string, int>();
            newMinMax.Add("min", val);
            newMinMax.Add("mmax", val);
            if(minMaxStack.Count>0){
                Dictionary<string, int> lastMinMax = new Dictionary<string, int>(minMaxStack[minMaxStack.Count-1]);
                newMinMax["min"] = Math.Min(lastMinMax["min"], val);
                newMinMax["max"] = Math.Max(lastMinMax["max"], val);                
            }
            minMaxStack.Add(newMinMax);
            stack.Add(val)            ;
        }
        //T:O(1) | S:O(1)
        public int GetMin(){
            return minMaxStack[minMaxStack.Count-1]["min"];
        }
        
        //T:O(1) | S:O(1)
        public int GetMax(){
            return minMaxStack[minMaxStack.Count-1]["max"];
        }        
        
    }
}