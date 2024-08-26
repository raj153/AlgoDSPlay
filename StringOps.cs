using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using System.Transactions;
using AlgoDSPlay.DataStructures;

namespace AlgoDSPlay
{
    public class StringOps
    {


      //https://www.algoexpert.io/questions/knuth-morris-pratt-algorithm KMP
      public static bool IsBigStringContainsSmallStringUsingKMP(string str, string substring){
        //T:O(n+m) | S:O(m)
        TODO:
        int[] pattern= BuildPattern(substring);
        return IsBigStringContainsSmallStringUsingKMP(str, substring, pattern);
      }

        private static bool IsBigStringContainsSmallStringUsingKMP(string str, string substring, int[] pattern)
        {
            int i=0, j=0;
            while(i+ substring.Length-j<=str.Length){

                if(str[i] == substring[j]){
                    if(j == substring.Length-1) return true;
                    i++; j++;
                }else if(j>0){
                    j= pattern[j-1]+1;
                }else i++;
            }
            return false;
        }

        private static int[] BuildPattern(string substring)
        {
            int[] pattern = new int[substring.Length];
            Array.Fill(pattern,-1);
            int j=0, i=1;

            while(i< substring.Length){
                if(substring[i] == substring[j]){
                    pattern[i] = j;
                    i++;
                    j++;
                }else if(j > 0){
                    j= pattern[j-1]+1;
                }else i++;

            }
            return pattern;
        }
        //https://www.algoexpert.io/questions/reverse-words-in-string
        public static string ReverseWordsInString(string str){
            
            //1. T:O(n) | S:O(n)
            string reversedStr = ReverseWordsInString1(str);

            //2. T:O(n) | S:O(n) -- Reverse entire string and re-reverse each word

            //3. T:O(n) | S:O(n) using Stack
            reversedStr = ReverseWordsInStringWithStack(str);
            
            return reversedStr;
        }

        private static string ReverseWordsInStringWithStack(string str)
        {
            Stack<string> reverseStack = new Stack<string>();
            while(str.Length >0){
                
                int idxSpace = str.IndexOf(" ");
                if(idxSpace >= 0){
                    string str1=str.Substring(0,idxSpace);
                    if(str1.Length != 0)
                        reverseStack.Push(str1);    
                    reverseStack.Push(" ");
                    str = str.Substring(idxSpace+1);   
                    //len=str.Length;                                                     
                }else {
                    reverseStack.Push(str);
                    str="";
                }                                
            }
            StringBuilder reverStr = new StringBuilder();
            while(reverseStack.Count >0){
                string s1 = reverseStack.Pop();
                reverStr.Append(s1);   

            }
            return reverStr.ToString();

        }

        private static string ReverseWordsInString1(string str)
        {
            List<string> words = new List<string>();
            int startOfWord=0;

            for(int idx=0; idx<str.Length; idx++){
                char c = str[idx];
                if( c == ' '){
                    words.Add(str.Substring(startOfWord, idx-startOfWord));
                    startOfWord = idx;
                }else if(str[startOfWord] == ' '){
                    words.Add(" ");
                    startOfWord = idx;
                }

            }
            words.Add(str.Substring(startOfWord));
            words.Reverse();
            return string.Join("", words);
        }
        //https://www.algoexpert.io/questions/semordnilap
        public static List<List<string>> Semordnilap(string[] words){

            //T:O(n*m) | S:O(n*m) where n represents total words and m is length of longest word
            HashSet<string> wordsSet = new HashSet<string>(words);
            List<List<string>> semoPairs = new List<List<string>>();

            foreach(var word in words){ // O(n)
                char[] chars = word.ToCharArray();
                Array.Reverse(chars); // O(m)
                string reverse = new string(chars);
                if(wordsSet.Contains(reverse) && (!word.Equals(reverse))){
                    List<string> semoPair = new List<string>{word, reverse};
                    semoPairs.Add(semoPair);
                    wordsSet.Remove(word);
                    wordsSet.Remove(reverse);

                }
            }
            return semoPairs;
        }
        //https://www.algoexpert.io/questions/levenshtein-distance
        //Minimum Edit operations(insert/delete/replace) required
        public static int LevenshteinDistiance(string str1, string str2){

            //1.Using full-blown DP(dyn pro) table 
            //T:O(nm) | S:O(nm)
            int numMinEditOps = LevenshteinDistiance1(str1, str2);

            //2.Using full-blown DP(dyn pro) table 
            //T:O(nm) | S:O(min(n,m))
            numMinEditOps = LevenshteinDistianceOptimal(str1, str2);

            return numMinEditOps;
        }

        private static int LevenshteinDistianceOptimal(string str1, string str2)
        {
            string small = str1.Length < str2.Length ? str1:str2;
            string big = str1.Length  >= str2.Length ? str2 : str1;

            int[] evenEdits = new int[small.Length+1];
            int[] oddEdits = new int[small.Length+1];
            for(int j=0; j< small.Length; j++){
                evenEdits[j]=j;
            }
            int[] currentEdit, prevEdit;
            for(int i=1; i<big.Length+1; i++){
                if(i%2 == 1){
                    currentEdit = oddEdits;
                    prevEdit = evenEdits;
                }else{
                    currentEdit = evenEdits;
                    prevEdit = oddEdits;
                }
                currentEdit[0]=i;
                for(int j=1; j< small.Length+1; j++){
                    if(big[j-1] == small[j-1]){
                        currentEdit[j]=prevEdit[j-1];                        
                    }else{
                        currentEdit[j] = 1+ Math.Min(prevEdit[j-1],
                        Math.Min(prevEdit[j], currentEdit[j-1]));
                    }
                }
            }
            return big.Length%2 == 0 ? evenEdits[small.Length] : oddEdits[small.Length];
        }

        private static int LevenshteinDistiance1(string str1, string str2)
        {
            int[,] edits = new int[str2.Length+1, str1.Length+1];
            for(int row=0; row < str2.Length+1; row++){
                for(int col=0; col < str1.Length+1; col++){
                    edits[row, col] = col;
                }
                edits[row,0]=row;
            }
            for(int row=1; row < str2.Length+1; row++){
                for(int col=1; col < str1.Length+1; col++){
                    if(str2[row-1] == str1[col-1]){
                        edits[row, col] = edits[row-1, col-1];
                    }else{
                        edits[row, col] = 1+ Math.Min(edits[row-1, col-1],
                                            Math.Min(edits[row-1, col], edits[row, col-1]));
                    }

                }
            }
            return edits[str2.Length, str1.Length];

        }
        //https://www.algoexpert.io/questions/underscorify-substring
        //Merge Intervals/ Overlap intervals
        public static string UnderscorifySubstring(string str, string substring){
            //Average case - T:O(n+m) | S:O(n) - n is length of main string and m is length of substring
            List<int[]> locations = MergeIntervals(GetLocations(str, substring));
            
            return Underscorify(str, locations);
        }

        private static string Underscorify(string str, List<int[]> locations)
        {
            int locationIdx=0, strIdx=0;
            bool inBetweenUnderscores = false;
            List<string> finalChars = new List<string>();
            int i=0;
            while(strIdx < str.Length && locationIdx < locations.Count){
                if(strIdx == locations[locationIdx][i]){
                    finalChars.Add("_");
                    
                    inBetweenUnderscores = !inBetweenUnderscores;
                    
                    if(!inBetweenUnderscores) locationIdx++;

                    i= i==1?0:1;
                }
                finalChars.Add(str[strIdx].ToString()); 
                strIdx+=1;
            }
            if(locationIdx < locations.Count){ // substring found at the end of main string
                finalChars.Add("_");
            }else if( strIdx < str.Length) //Adding remaining main string
                finalChars.Add(str.Substring(strIdx));
            
            return String.Join("",finalChars);
            
        }

        private static List<int[]> MergeIntervals( List<int[]> locations)
        {
            if(locations.Count ==0) return locations;

            List<int[]> newLocations = new List<int[]>();
            newLocations.Add(locations[0]);
            int[] previous = newLocations[0];
            for(int i=1; i< locations.Count; i++){
                int[] current= locations[i];
                if(current[0]<=previous[1]) //Overlap check
                    previous[1]=current[1];
                else{
                    newLocations.Add(current);
                    previous = current;
                }
            }
            return newLocations;
        }

        private static  List<int[]> GetLocations(string str, string substring)
        {
            List<int[]> locations = new List<int[]>();
            int startIdx =0;
            while(startIdx < str.Length){
                int nextIdx = str.IndexOf(substring, startIdx); //O(n+m)
                if(nextIdx != -1){
                    locations.Add(new int[]{nextIdx, nextIdx+substring.Length});
                    startIdx = nextIdx+1;
                }else{
                    break;
                }
            }
            return locations;
        }
    }
}