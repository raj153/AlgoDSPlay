using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AlgoDSPlay.DataStructures;

namespace AlgoDSPlay
{
    public class TrieProbs
    {
        //https://www.algoexpert.io/questions/strings-made-up-of-strings
        public static string[] StringMadeUpOfSubstrings(string[] strings, string[] substrings){
            //T:O(s2*m+s1*n^2) | S:O(s2*m+s1*n) - where s2 is number of substrings and m is length of longest substring.
            //s1 is number of strings and n is legnth of longest string
            Trie trie = new Trie();

            foreach(string str in substrings){
                trie.Insert(str);
            }

            List<string> solutions = new List<string>();
            foreach(var str in strings){
                if(IsMadeUpOfSubstrings(str, 0, trie, new Dictionary<int, bool>())){
                    solutions.Add(str);
                }
            }
            return solutions.ToArray();
        }

        private static bool IsMadeUpOfSubstrings(string str, int startIdx, Trie trie, Dictionary<int, bool> memo)
        {
            if(startIdx == str.Length) return true; //Base case
            
            if(memo.ContainsKey(startIdx)) return memo[startIdx]; //Base case

            TrieNode currentTrieNode = trie.Root;
            
            for(int currentCharIdx=startIdx; currentCharIdx < str.Length; currentCharIdx++){
                
                char curChar = str[currentCharIdx];

                if(!currentTrieNode.Children.ContainsKey(curChar)) break;

                currentTrieNode = currentTrieNode.Children[curChar];
                if(currentTrieNode.IsEndOfString && IsMadeUpOfSubstrings(str, currentCharIdx+1, trie, memo)){
                    memo[startIdx] =true;
                    return true;
                }
            }
            memo[startIdx] = false;
            return false;

        }
             //https://www.algoexpert.io/questions/boggle-board
        public static List<string> BoggleBoard(char[,] board, string[] words){
            //T:O(WS+MN*8^S) | S:O(WS+MN)
            Trie trie = new Trie();
            foreach(var word in words){
                trie.Insert(word);
            }
            HashSet<string> finalWords = new HashSet<string>();
            bool[,] visited = new bool[board.GetLength(0), board.GetLength(1)];
            for(int row=0; row< board.GetLength(0); row++){
                for(int col=0; col < board.GetLength(1); col++){
                    Explore(row, col, board, trie.Root, visited, finalWords);
                }
            }
            List<string> finalWordsArray = new List<string>();
            foreach(string key in finalWords){
                finalWordsArray.Add(key);
            }
            return finalWordsArray;

        }

        private static void Explore(int row, int col, char[,] board, TrieNode trieNode, bool[,] visited, HashSet<string> finalWords)
        {
            if(visited[row, col]) return;

            char letter = board[row, col];
            if(!trieNode.Children.ContainsKey(letter)) return;
            visited[row, col]= true;

            trieNode = trieNode.Children[letter];
            if(trieNode.Children.ContainsKey('*')){  //endSymbol checking
                finalWords.Add(trieNode.Word);
            }

            List<int[]> neighbors = GetNeighbors(row, col,board);
            foreach(int[] neighbor in neighbors){
                Explore(neighbor[0], neighbor[1], board, trieNode,visited, finalWords);
            }
            visited[row, col]=false;

        }

        public static List<int[]> GetNeighbors(int row, int col, char[,] board){
            
            List<int[]> neighbors = new List<int[]>();

            //Top-Left Diagonal 
            if(row >0 && col > 0)
                neighbors.Add(new int[]{row-1, col-1});
            //Top
            if(row >0)
                neighbors.Add(new int[]{row-1, col});
            //Top-Right Diagonal
            if(row >0 && col < board.GetLength(1)-1)
                neighbors.Add(new int[]{row-1, col+1});
            //Right
            if(col < board.GetLength(1)-1)
                neighbors.Add(new int[]{row, col+1});
            //Down-Right Diagonal
            if(row > board.GetLength(0)-1 && col < board.GetLength(1)-1)
                neighbors.Add(new int[]{row+1, col+1});
            //Down
            if(row > board.GetLength(0)-1)
                neighbors.Add(new int[]{row+1, col});
            //Down-Left Diagonal
            if(row > board.GetLength(0)-1 && col >0)
                neighbors.Add(new int[]{row+1, col-1});
            //Left
            if(col>0)
                neighbors.Add(new int[]{row, col-1});

            return neighbors;
        }

         //https://www.algoexpert.io/questions/multi-string-search
        public static List<bool> MultiStringSearch(string bigString, string[] smallStrings){
            List<bool> solution = new List<bool>();
            //1.Naive - T:O(bns) | S:O(n) where b is length of big string, s is length of biggest small string and n is length of small string array
            foreach(string smallString in smallStrings){
                solution.Add(IsInBigString(bigString, smallString));

            }
            //2.ModifiedSuffixTrie - T:O(b^2+ns) | S:O(b^2+n)
            ModifiedSuffixTrie modifiedSuffixTrie = new ModifiedSuffixTrie(bigString);
            solution.Clear();
            foreach(string smallString in smallStrings){
                solution.Add(modifiedSuffixTrie.Contains(smallString));
            }

            solution.Clear();            
            //3.Trie - T:O(ns+bs) | S:O(ns)
            Trie trie = new Trie();
            foreach(string smallString in smallStrings){
                trie.Insert(smallString);
            }
            HashSet<string> containedStrings = new HashSet<string>();
            for(int i=0; i<bigString.Length; ++i){
                FindSmallStringsIn(bigString, i, trie, containedStrings);
            }
            foreach(string smallString in smallStrings){
                solution.Add(containedStrings.Contains(smallString));
            }
            return solution;

        }

        private static void FindSmallStringsIn(string bigString, int startIdx, Trie trie, HashSet<string> containedStrings)
        {
            TrieNode currentNode = trie.Root;
            for(int i=startIdx; i<bigString.Length; ++i){
                char currentChar = bigString[i];
                if(!currentNode.Children.ContainsKey(currentChar)){
                    break;
                }
                currentNode = currentNode.Children[currentChar];
                if(currentNode.Children.ContainsKey(trie.EndSymbol)){
                    containedStrings.Add(currentNode.Word);
                }
            }

            
        }

        private static bool IsInBigString(string bigString, string smallString)
        {
            for(int i=0; i< bigString.Length; ++i){
                if(i+ smallString.Length > bigString.Length){
                    break;
                }
                if(IsInBigString(bigString, smallString, i)){
                    return true;
                }
            }
            return false;
        }

        private static bool IsInBigString (string bigString, string smallString, int startIdx)
        {
            //big bigger
            //egg
            int leftBigIndex= startIdx;
            int rightBigIndex= startIdx+ smallString.Length-1;
            int leftSmallIndex = 0;
            int rightSmallIndex=smallString.Length-1;
            while(leftBigIndex <= rightBigIndex){

                if(bigString[leftBigIndex] != smallString[leftSmallIndex]
                    || bigString[rightBigIndex] != smallString[rightSmallIndex])
                    return false;
                
                leftBigIndex++;
                rightSmallIndex--;
                leftSmallIndex++;
                rightBigIndex--;
            } 
            return true;
 
      }
      //https://www.algoexpert.io/questions/longest-most-frequent-prefix
      public static string LongestMostFrequentPrefix(string[] strings){

        //T:O(n*m) | S:O(n*m) where n is number of strings and m is length of longest string in strings array
        Trie trie = new Trie();
        foreach(string str in strings){
            trie.Insert(str);
        }
        return trie.MaxPrefixFullString.Substring(0, trie.MaxPrefixLen);
        
      }
    }
}