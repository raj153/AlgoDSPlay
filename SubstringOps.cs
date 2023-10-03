using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay
{
    public class SubstringOps
    {
        //https://www.algoexpert.io/questions/smallest-substring-containing
        public static string SmallestSubstringContaining(string bigStr,string smallStr){
            //T:O(b+s) | S:O(b+s) where b is length of big string and s is length of small string

            Dictionary<char, int> targetCharCounts = GetCharCounts(smallStr);
            List<int> subStringBounds = GetSubstringBounds(bigStr, targetCharCounts);
            return GetStringFromBounds(bigStr, targetCharCounts);
        }

        private static string GetStringFromBounds(string bigStr, Dictionary<char, int> targetCharCounts)
        {
            throw new NotImplementedException();
        }

        private static List<int> GetSubstringBounds(string str, Dictionary<char, int> targetCharCounts)
        {
            List<int> substringBounds = new List<int>{0,Int32.MaxValue};

            Dictionary<char, int> substringCharCounts = new Dictionary<char, int>();
            int numUniqueChars = targetCharCounts.Count;
            int numUniqueCharsDone = 0;
            int leftIdx=0, rightIdx=0;

            while(rightIdx < str.Length){
                char rightChar = str[rightIdx];

                if(!targetCharCounts.ContainsKey(rightChar)){
                    rightIdx++;
                    continue;
                }
                if(!substringCharCounts.ContainsKey(rightChar))
                    substringCharCounts[rightChar]=0;

                substringCharCounts[rightChar]++;    

                if (substringCharCounts[rightChar] == targetCharCounts[rightChar]){
                    numUniqueCharsDone++;
                }

                while(numUniqueCharsDone == numUniqueChars && leftIdx <= rightIdx){

                //substringBounds = GetCloserBound 
                // TODO:

                }
                
            }
            return substringBounds;

        }

        private static Dictionary<char, int> GetCharCounts(string smallStr)
        {
             Dictionary<char, int> charCounts = new Dictionary<char, int>();

             foreach(char c in smallStr){
                if(!charCounts.ContainsKey(c))
                    charCounts[c]=0;
                charCounts[c]++;
             }

             return charCounts;
        }

        //https://www.algoexpert.io/questions/longest-substring-without-duplication
        public static string LongestSubstringWithoutDuplication(string str){
            //T:O(n) | S:O(min(n,a)) where n is length of input string and a is length of unique letters  represented in the input string
            Dictionary<char, int> lastSeen = new Dictionary<char, int>();
            int startIndex=0;
            int subStrLen=1;
            lastSeen[str[0]]=0;
            int[] longest = {0, 1};
            for(int curIndex=1; curIndex<str.Length; curIndex++){
                
                if(lastSeen.ContainsKey(str[curIndex])){
                    
                    startIndex=  Math.Max(startIndex, lastSeen[str[curIndex]]+1);
                    lastSeen[str[curIndex]]=curIndex;                    
                }
                  
                int curSubstrLen = curIndex-startIndex+1;                    
                if(curSubstrLen > subStrLen){
                    subStrLen =curSubstrLen;                        
                    longest[0]=startIndex;
                    longest[1] = curIndex;
                }
                lastSeen[str[curIndex]]=curIndex;
                   
                
            }
            return str.Substring(longest[0], longest[1]-longest[0]+1) ;

        }



    }
}