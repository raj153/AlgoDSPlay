using System.Collections;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay
{
    public class StackProbs
    {
        //https://www.algoexpert.io/questions/balanced-brackets
        public static bool BalancedBrackets(string str){
            //T:O(n)|S:O(n)
            string openingBrackets ="([{";
            string closingBrackets = ")]}";
            Dictionary<char, char> matchingBrackets = new Dictionary<char, char>();
            matchingBrackets.Add('(',')');
            matchingBrackets.Add('[',']');
            matchingBrackets.Add('{','}');
            List<char> stack = new List<char>();
            for(int i=0; i< str.Length; i++){
                char letter = str[i];
                if(openingBrackets.IndexOf(letter) != -1){
                    stack.Add(letter);                    
                }else if(closingBrackets.IndexOf(letter) != -1){
                    if(stack.Count ==0) return false;

                    if (stack[stack.Count-1] == matchingBrackets[letter]){
                        stack.RemoveAt(stack.Count-1);
                    }else{
                        return false;
                    }
                }
            }
            //Using Stack - 
            /*
            Stack<char> stack = new Stack<char>();
            foreach(char c in str){

                if(openBrak.IndexOf(c) !=-1)
                        stack.Push(c);
                else if(closingBrak.IndexOf(c) != -1)
                {
                    if(stack.Count == 0) return false;

                    if(matchBrak[stack.Peek()] != c) return false;
                    stack.Pop();                      
                }          
            }
            */
            return stack.Count == 0;
        }

        //https://www.algoexpert.io/questions/colliding-asteroids
        public static int[] CollidingAsteroids(int[] asteroids) {
            //T:O(n) | S:O(n)
            if(asteroids.Length ==0) return new int[]{};
            
            Stack<int> asters = new Stack<int>();
            
            foreach(int aster in asteroids){
                if(asters.Count ==0 || aster > 0 || asters.Peek()<0){
                    asters.Push(aster);
                    continue;
                }                
                while(asters.Count >0){
                    if(asters.Peek() < 0){
                        asters.Push(aster);
                        break;
                    }
                    //-3,5,-8
                    int prevAster = asters.Peek();                    
                    
                    if( prevAster > Math.Abs(aster)) { break;}
                    if(prevAster == Math.Abs(aster)) { asters.Pop(); break;}

                    asters.Pop();                    

                    if(asters.Count == 0){
                        asters.Push(aster);
                        break;
                    }

                } 
                
            }
            int[] res= new int[asters.Count()];
            for(int i=asters.Count-1; i>=0; i--){
                res[i]=asters.Pop();
            }
            return res;
        }
        //https://www.algoexpert.io/questions/longest-balanced-substring
        public int LongestBalancedSubstring(string str){
            int maxLen=0;
            
            //1. Naive/Bruteforce - Pair of loops and Stack
            //T:O(n^3) | S:O(n)
            maxLen = LongestBalancedSubstringNaive(str);
            
            //2. Optimal with Stack space
            //T:O(n) | S:O(n)
            maxLen = LongestBalancedSubstringOptimal1(str);

            //3. Optimal with NO auxiliary space
            //T:O(n) | S:O(1)
            maxLen = LongestBalancedSubstringOptimal2(str);

            //4. Optimal with NO auxiliary space, simplified version of #3
            //T:O(n) | S:O(1)
            maxLen = LongestBalancedSubstringOptimal3(str);

            return maxLen;
        }

        private int LongestBalancedSubstringOptimal3(string str)
        {
            return Math.Max(GetLongestBalancedDirection(str, true), GetLongestBalancedDirection(str, false));
        }

        private int GetLongestBalancedDirection(string str, bool leftToRight)
        {
            char openingParens = leftToRight? '(':')';
            int strIdx = leftToRight ? 0 : str.Length-1;
            int step=leftToRight?1:-1;

            int maxLen =0;
            int openingCount=0, closingCount =0;

            int idx = strIdx;
            while(idx >=0 && idx <str.Length)
            {
                char c = str[idx];

                if(c == openingParens) openingCount++;
                else closingCount++;

                if(openingCount == closingCount)
                    maxLen = Math.Max(maxLen, closingCount*2);
                else if (closingCount > openingCount){
                    openingCount =0; 
                    closingCount=0;
                }
                idx+=step;
            }
            return maxLen;
        }

        private int LongestBalancedSubstringOptimal2(string str)
        {
            int maxLen =0;
            int openingCount =0, closingCount=0;
            
            for(int i=0; i<str.Length; i++){
                char c = str[i];

                if( c == '('){
                    openingCount +=1;
                }else closingCount+=1;

                if(openingCount == closingCount)
                    maxLen = Math.Max(maxLen, closingCount*2);
                else if ( closingCount > openingCount){
                    openingCount =0;
                    closingCount =0;
                }
            }
            openingCount =0;
            closingCount =0;
            //scenario: ((())( where opening brackets are more than closing one and still a valid substring exists
            for(int i=str.Length-1; i>=0; i--){
                char c = str[i];

                if(c == '('){
                    openingCount++;
                }else closingCount++;

                if(openingCount == closingCount) 
                    maxLen = Math.Max(maxLen, openingCount*2);
                else if (openingCount > closingCount){
                    openingCount =0;
                    closingCount =0;
                }
            }
            return maxLen;
        }

        private int LongestBalancedSubstringOptimal1(string str)
        {
            int maxLen=0;
            Stack<int> idxStack = new Stack<int>();
            idxStack.Push(-1);

            for(int i=0; i<str.Length; i++){
                if(str[i] == '('){
                    idxStack.Push(i);
                }else {
                    idxStack.Pop(); //-1 is there by-default
                    if(idxStack.Count ==0)
                        idxStack.Push(i);                    
                    else{
                        int balancedSubstringStartIdx = idxStack.Peek();
                        int currentLen = i - balancedSubstringStartIdx;
                        maxLen = Math.Max(maxLen, currentLen);
                    }
                }
            }
            return maxLen;
        }

        private int LongestBalancedSubstringNaive(string str)
        {
            int maxLen=0;
            for(int i=0; i< str.Length; i++){
                for(int j=i+2; j<str.Length+1; j++){
                    if(IsBalanced(str.Substring(i, j-i))){
                        int currentLen = j-i;
                        maxLen= Math.Max(currentLen, maxLen);
                    }
                }
            }
            return maxLen;
        }

        private bool IsBalanced(string str)
        {
            Stack<char> openParamsStack = new Stack<char>();

            for(int i=0; i< str.Length; i++){
                char c = str[i];
                if( c == '('){
                    openParamsStack.Push('(');
                }else if( openParamsStack.Count >0){
                    openParamsStack.Pop();
                }else{
                    return false;
                }
            }
            return openParamsStack.Count ==0;
        }
        
        
    }
}