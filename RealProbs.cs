using System.Text;
using System.Runtime.InteropServices;
using System.Globalization;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.Security.Cryptography.X509Certificates;
using System.Threading.Tasks.Dataflow;
using Microsoft.VisualBasic;
using AlgoDSPlay.DataStructures;
using System.Security.Cryptography;
using System.Net.Http.Headers;
using System.Linq.Expressions;

namespace AlgoDSPlay
{
    public class RealProbs
    {


        //https://www.algoexpert.io/questions/run-length-encoding
        public static string RunLengthEncoding(string str){
            //T:O(n)|S:O(n)
            StringBuilder encodedStringChars = new StringBuilder();
            int currentRunLength=1;

            for(int i=1; i < str.Length; i++){
                char currentChar = str[i];
                char previousChar = str[i-1];

                if((currentChar != previousChar) || (currentRunLength ==9)){
                    encodedStringChars.Append(currentRunLength.ToString());
                    encodedStringChars.Append(previousChar);
                    currentRunLength =0;
                }
                currentRunLength +=1;
            }   
            
            return encodedStringChars.ToString();
        }   
        //https://www.algoexpert.io/questions/tandem-bicycle
        public int TandemCycle(int[] redShirtSpeeds, int[] blueShirtSpeeds, bool fastest){
            //T: O(nlog(n))| S:O(1)
            Array.Sort(redShirtSpeeds);
            Array.Sort(blueShirtSpeeds);

            if(fastest)
                reverseArrayInPlace(redShirtSpeeds);
            
            int totalSpeed =0; 
            for(int idx=0; idx< redShirtSpeeds.Length; idx++){
                int rider1 = redShirtSpeeds[idx];
                int rider2 = blueShirtSpeeds[idx];
                totalSpeed += Math.Max(rider1, rider2);
            }
            return totalSpeed;

        }

        private void reverseArrayInPlace(int[] array)
        {
            int start=0;
            int end= array.Length-1;
            while(start < end){
                int temp = array[start];
                array[start] = array[end];
                array[end]= temp;
                start++;
                end--;
            }
        }

        //https://www.algoexpert.io/questions/class-photos
        public static bool CanTakeClassPhotos(List<int> redShirtHeights, List<int> blueShirtHeights){

            //T: O(Nlog(N)) | S: O(1)
            redShirtHeights.Sort((a,b)=> b.CompareTo(a));
            blueShirtHeights.Sort((a,b)=> b.CompareTo(a));

            string shirtColorInFirstRow = (redShirtHeights[0] < blueShirtHeights[0]? "RED": "BLUE");

            for(int idx=0; idx < redShirtHeights.Count; idx++){
                int redShirtHeight = redShirtHeights[idx];
                int blueShirtHeight = blueShirtHeights[idx];

                if(shirtColorInFirstRow == "RED")
                    if(redShirtHeight >= blueShirtHeight) return false;
                else{
                    if(blueShirtHeight >= redShirtHeight) return false;
                }

            }
            return true;


        }
        //https://www.algoexpert.io/questions/calendar-matching
        public static List<StringMeeting> CalendarMatching(List<StringMeeting> calendar1, StringMeeting dailyBounds1,
                                                           List<StringMeeting> calendar2, StringMeeting dailyBounds2,
                                                           int meetringDuration
                                                           )
        {
            //T:O(c1+c2) | S:O(c1+c2)
            List<Meeting> updateCalendar1 = UpdateCalendar(calendar1, dailyBounds1);
            List<Meeting> updateCalendar2 = UpdateCalendar(calendar2, dailyBounds2);
            List<Meeting> mergedCalendar = MergeCalendars(updateCalendar1, updateCalendar2);
            List<Meeting> mergeIntervals =  MergeIntervals(mergedCalendar); 

            return GetMatchingAvailabilities(mergeIntervals, meetringDuration);
        }

        private static List<StringMeeting> GetMatchingAvailabilities(List<Meeting> mergeIntervals, int meetringDuration)
        {
            List<StringMeeting> availableMeetingSlots = new List<StringMeeting>();

            for(int i=1; i< mergeIntervals.Count; i++){
                int start = mergeIntervals[i-1].End;
                int end = mergeIntervals[i].Start;
                int availabilityDuration = end-start; 
                if(availabilityDuration >= meetringDuration)
                    availableMeetingSlots.Add(new StringMeeting(MinutesToTime(start), MinutesToTime(end)));

            }
            return availableMeetingSlots;
        }

        private static List<Meeting> MergeIntervals(List<Meeting> calendar)
        {
            List<Meeting> mergedIntervals = new List<Meeting>();
            mergedIntervals.Add(calendar[0]);
            for(int i=1; i<calendar.Count; i++){
                Meeting currentMeeting = calendar[i];
                Meeting previousMeeting = mergedIntervals[mergedIntervals.Count -1];
                //6-11 8-10
                if(currentMeeting.Start <= previousMeeting.End)
                    mergedIntervals[mergedIntervals.Count -1].End = Math.Max(currentMeeting.End, previousMeeting.End);
                else 
                    mergedIntervals.Add(currentMeeting); 
            }
            return mergedIntervals;
        }

        private static List<Meeting> MergeCalendars(List<Meeting> calendar1, List<Meeting> calendar2)
        {
            List<Meeting> merged = new List<Meeting>();
            
            int i=0, j=0;
            while(i < calendar1.Count && j < calendar2.Count){
                Meeting meeting1 = calendar1[i];
                Meeting meeting2 = calendar2[j];

                if(meeting1.Start < meeting2.Start){
                    merged.Add(meeting1);
                    i++;
                }else{
                    merged.Add(meeting2);
                    j++;
                }
            }
            while(i<calendar1.Count) merged.Add(calendar1[i]);
            while(j< calendar2.Count) merged.Add(calendar2[j]);

            return merged;

        }

        private static List<Meeting> UpdateCalendar(List<StringMeeting> calendar, StringMeeting dailyBounds)
        {
            List<StringMeeting> updatedCalendar = new List<StringMeeting>();
            updatedCalendar.Add(new StringMeeting("0:00", dailyBounds.Start));
            updatedCalendar.AddRange(calendar);
            updatedCalendar.Add(new StringMeeting(dailyBounds.End, "23:59"));
            List<Meeting> calendarInMins = new List<Meeting>();
            for(int i=0; i< updatedCalendar.Count; i++){
                calendarInMins.Add(new Meeting(TimeToMinuts(updatedCalendar[i].Start), TimeToMinuts(updatedCalendar[i].End) ));
            }
            return calendarInMins;
        }
        private static int TimeToMinuts(string time)
        {
            string[] delimProps = time.Split(':');
            int hours = Int32.Parse(delimProps[0]);
            int minutes = Int32.Parse(delimProps[1]);
            return hours*60+minutes;

            
        }
        private static string MinutesToTime(int minutes){
            int hours= minutes/60;
            int mins= minutes%60;
            return hours.ToString()+":"+ (mins< 10 ? "0"+mins.ToString():mins.ToString());
        }
    
        //https://www.algoexpert.io/questions/generate-div-tags
        public static List<string> GenerateDivTags(int numberOfTags){
            // O((2n)!/((n!((n + 1)!)))) time | O((2n)!/((n!((n + 1)!)))) space -
            // where n is the input number
            List<string> matchedDivTags = new List<string>();
            GenerateDivTagsFromPrefix(numberOfTags, numberOfTags, "", matchedDivTags) ;
            return matchedDivTags;
        }

        private static void GenerateDivTagsFromPrefix(int openingTagsNeeded, int closingTagsNeeded, string prefix, List<string> result)
        {
            if(openingTagsNeeded >0){
                string newPrefix = prefix+"<div>";
                GenerateDivTagsFromPrefix(openingTagsNeeded-1, closingTagsNeeded, newPrefix, result);
            }

            if(openingTagsNeeded < closingTagsNeeded) {
                string newPrefix =prefix +"</div>";
                GenerateDivTagsFromPrefix(openingTagsNeeded, closingTagsNeeded-1, newPrefix,result);
            }
            if(closingTagsNeeded == 0)
            result.Add(prefix);
        }
        //https://www.algoexpert.io/questions/best-seat
        public static int BestSeat(int[] seats) {
            //T:O(n) | S:O(1)
            int bestSeat=-1, maxSpace=0;
            
            int left=0; ;
            while(left <seats.Length-1){
                int right=left+1;
                while(right < seats.Length && seats[right] ==0)
                    right++;
                
                int availableSpace = right-left+1;
                if(availableSpace > maxSpace){
                    bestSeat =(left+right)/2;
                    maxSpace = availableSpace;
                }
                left=right;

            }
            return bestSeat;
        }
        //https://www.algoexpert.io/questions/number-of-ways-to-make-change 
        //Coins
        public static int NumberOfWaysToMakeChange(int n, int[] denoms ){
            
            //T: O(nd) | S: O(n) where d denotes denominations and n denotes amounts
            int[] ways = new int[n+1];
            ways[0]=1;
            
            foreach(int denom in denoms){
                for(int amount=0; amount< n+1;amount++){
                    if(denom <= amount){
                        ways[amount] += ways[amount-denom];
                    }

                }
            }
            return ways[n];
        }
        //https://www.algoexpert.io/questions/min-number-of-coins-for-change
        public static int MinimumNumberOfCoinsForChange(int n, int[] denoms){
            //T:O(n*d) | S:O(n)
            int[] numOfCoins = new int[n+1];
            Array.Fill(numOfCoins, Int32.MaxValue);
            numOfCoins[0]=0;
            int toCompare =0;
            foreach(int denom in denoms){
                for(int amount=0; amount <=numOfCoins.Length; amount++){
                    if(denom<=amount){
                        if(numOfCoins[amount-denom] == Int32.MaxValue){
                            toCompare = numOfCoins[amount-denom];
                        }
                        else{
                            toCompare = numOfCoins[amount-denom]+1;
                        }
                        numOfCoins[amount]=Math.Min(numOfCoins[amount], toCompare);
                }
            }
            }
            return numOfCoins[n]!= int.MaxValue ? numOfCoins[n] : -1;
        
        }
        //https://www.algoexpert.io/questions/pattern-matcher
        public static string[] PatternMatacher(string pattern, string str){
            //T:O(n^2+m) | S:O(n+m)
            if(pattern.Length > str.Length)
                return new string[]{};
            
            char[] newPattern = GetNewPattern(pattern);
            bool didSwitch = newPattern[0] != pattern[0];
            Dictionary<char, int> counts = new Dictionary<char, int>();
            counts['x']=0;
            counts['y']=0;
            int firstYPos = GetCountsAndFirstYPos(newPattern, counts);
            if(counts['y']!=0){
                for(int lenOfX=1; lenOfX < str.Length; lenOfX++){
                    double lenOfY = (double)(str.Length-lenOfX*counts['x'])/counts['y'];
                    
                    if(lenOfY <=0 || lenOfY%1 !=0)
                        continue;
                    
                    int yIdx = firstYPos* lenOfX;
                    string x =str.Substring(0,lenOfX);
                    string y = str.Substring(yIdx,(int)lenOfY);
                    string potentialMatch = BuildPotentialMatch(newPattern, x,y);
                    if(str.Equals(potentialMatch)){
                        return didSwitch? new string[]{y,x}: new string[]{x,y};
                    }
                }
            }else{
                double lenOfX = str.Length/counts['x'];
                if(lenOfX % 1 ==0){
                    string x= str.Substring(0, (int) lenOfX);
                    string potentialMatch = BuildPotentialMatch(newPattern, x, "");
                    if(str.Equals(potentialMatch)){
                        return didSwitch? new string[]{"",x}: new string[]{x,""};
                    }
                }

            }
            return new string[]{};
        }

        private static string BuildPotentialMatch(char[] newPattern, string x, string y)
        {
            StringBuilder potentialMatch = new StringBuilder();
            foreach(char c in newPattern){
                if(c == 'x'){
                    potentialMatch.Append(x);
                }else potentialMatch.Append(y);
            }
            return potentialMatch.ToString();
        }

        private static int GetCountsAndFirstYPos(char[] pattern, Dictionary<char, int> counts)
        {
            int firstYPos = -1;
            for(int i=0; i< pattern.Length; i++){
                counts[pattern[i]]++;
                if(pattern[i] == 'Y' && firstYPos == -1){
                    firstYPos = i;
                }
            }
            return firstYPos;
        }

        private static char[] GetNewPattern(string pattern)
        {
            char[] patternLetters = pattern.ToCharArray();
            if(pattern[0] == 'x') return patternLetters;
            for(int i=0; i< patternLetters.Length; i++){
                if(patternLetters[i]=='x')
                    patternLetters[i]='y';
                else
                    patternLetters[i]='x';

            }
            return patternLetters;
        }

        public class Meeting{
            public int Start{get;set;}
            public int End {get;set;}

            public Meeting(int start, int end){
                this.Start= start;
                this.End = end;
            }
        }
        public class StringMeeting
        {
            public string Start{get;set;}
            public string End{get;set;}

            public StringMeeting(string start, string end){
                this.Start = start;
                this.End = end;
            }
        }
        //https://www.algoexpert.io/questions/search-for-range
          // O(log(n)) time | O(log(n)) space
        public static int[] SearchForRangeNonOptimal(int[] array, int target) {
            int[] finalRange = { -1, -1 };
            alteredBinarySearchNonOptimal(array, target, 0, array.Length - 1, finalRange, true);
            alteredBinarySearchNonOptimal(array, target, 0, array.Length - 1, finalRange, false);
            return finalRange;
        }

        public static void alteredBinarySearchNonOptimal(
            int[] array, int target, int left, int right, int[] finalRange, bool goLeft
        ) {
            if (left > right) {
            return;
            }
            int mid = (left + right) / 2;
            if (array[mid] < target) {
            alteredBinarySearchNonOptimal(array, target, mid + 1, right, finalRange, goLeft);
            } else if (array[mid] > target) {
            alteredBinarySearchNonOptimal(array, target, left, mid - 1, finalRange, goLeft);
            } else {
            if (goLeft) {
                if (mid == 0 || array[mid - 1] != target) {
                finalRange[0] = mid;
                } else {
                alteredBinarySearchNonOptimal(array, target, left, mid - 1, finalRange, goLeft);
                }
            } else {
                if (mid == array.Length - 1 || array[mid + 1] != target) {
                finalRange[1] = mid;
                } else {
                alteredBinarySearchNonOptimal(
                    array, target, mid + 1, right, finalRange, goLeft
                );
                }
            }
            }
        }

        // O(log(n)) time | O(1) space
        public static int[] SearchForRangeOptimal(int[] array, int target) {
            int[] finalRange = { -1, -1 };
            alteredBinarySearch(array, target, 0, array.Length - 1, finalRange, true);
            alteredBinarySearch(array, target, 0, array.Length - 1, finalRange, false);
            return finalRange;
        }

        public static void alteredBinarySearchOptimal(
            int[] array, int target, int left, int right, int[] finalRange, bool goLeft
        ) {
            while (left <= right) {
            int mid = (left + right) / 2;
            if (array[mid] < target) {
                left = mid + 1;
            } else if (array[mid] > target) {
                right = mid - 1;
            } else {
                if (goLeft) {
                if (mid == 0 || array[mid - 1] != target) {
                    finalRange[0] = mid;
                    return;
                } else {
                    right = mid - 1;
                }
                } else {
                if (mid == array.Length - 1 || array[mid + 1] != target) {
                    finalRange[1] = mid;
                    return;
                } else {
                    left = mid + 1;
                }
                }
            }
            }
        }
        //https://www.algoexpert.io/questions/dice-throws
        public static int DiceThrows(int numSides, int numDices, int target){

            //1.Naive Recursive - T:O(d*s*t) | S:O(d*t) - Repetive calculations due to overlap
            
            //2.Optimal - Recursive- Storing intermediate results 
            //T:O(d*s*t) | S:O(d*t)
            int numWaysToReachTarget = DiceThrowsRecursive(numSides, numDices, target);

            //3.Optimal - Iterative- Storing intermediate results 
            //T:O(d*s*t) | S:O(d*t)
             numWaysToReachTarget = DiceThrowsIterative(numSides, numDices, target);

            //3.Optimal at space - Iterative- Storing intermediate results for just two rows, previous and new row. 
            //T:O(d*s*t) | S:O(t) 
             numWaysToReachTarget = DiceThrowsIterativeSpaceOptimal(numSides, numDices, target);

            return numWaysToReachTarget;
        }

        private static int DiceThrowsIterativeSpaceOptimal(int numSides, int numDices, int target)
        {
            int[,] storedResults = new int[2, target+1];
            storedResults[0,0]=1;

            int prevousNumDiceIndex =0;
            int newNumDiceIndex = 1;
            for(int currentNumDice=0; currentNumDice < numDices; currentNumDice++){
                for(int currentTarget =0; currentTarget <=target; currentTarget++){
                    int numWaysToReachTarget =0;
                    for(int currentNumSides =1;
                            currentNumSides <=Math.Min(currentTarget, numSides);
                            currentNumSides++){
                        
                        numWaysToReachTarget += storedResults[prevousNumDiceIndex, currentTarget-currentNumSides];

                    }                        
                    storedResults[newNumDiceIndex, currentTarget] = numWaysToReachTarget;
                }
                int tempPrevNumDiceIndex = prevousNumDiceIndex;
                prevousNumDiceIndex = newNumDiceIndex;
                newNumDiceIndex = tempPrevNumDiceIndex;
            }
            return storedResults[prevousNumDiceIndex, target];
        }

        private static int DiceThrowsIterative(int numSides, int numDices, int target)
        {
           int[,] storedResults = new int[numSides+1, target+1];
           storedResults[0,0] = 1;

           for(int currentNumDice =1; currentNumDice <=numDices;currentNumDice++ ){
            for(int currentTarget =0; currentTarget <=target; currentTarget++){
                int numWaysToReachTarget =0;
                for(int currentNumSides=1; currentNumSides <= Math.Min(currentTarget, numSides); 
                        currentNumSides++){
                            numWaysToReachTarget += storedResults[currentNumDice-1, currentTarget-currentNumSides];
                        }
                        storedResults[currentNumDice, currentTarget] = numWaysToReachTarget;
            }
           }
           return storedResults[numDices, target];
        }

        private static int DiceThrowsRecursive(int numSides, int numDices, int target)
        {
            int[,] storedResults = new int[numSides+1, target+1];

            for(int row=0; row < storedResults.GetLength(0); row++){
                for(int col=0; col< storedResults.GetLength(1); col++){
                    storedResults[row, col]=-1;
                }
            }
            return DiceThrowsRecursiveHelper(numDices, numSides, target, storedResults);
        }

        private static int DiceThrowsRecursiveHelper(int numDices, int numSides, int target, int[,] storedResults)
        {
            if(numDices == 0 ){ //DT(0,0)
                return target == 0 ?1 :0;
            }
            if(storedResults[numDices, target] != -1)
            {
                return storedResults[numDices, target];
            }
            int numWaysToReachTarget =0;
            for(int currentTarget = Math.Max(0, target-numSides);
                    currentTarget < target; currentTarget++){
                
                numWaysToReachTarget += DiceThrowsRecursiveHelper(numDices-1, numSides, currentTarget, storedResults);
            }
            storedResults[numDices, target] = numWaysToReachTarget;
            return numWaysToReachTarget;
        }
        //https://www.algoexpert.io/questions/disk-stacking
        public static List<int[]> DiskStacking(List<int[]> disks){
            //1.T:O(n^2) | S:O(n)
            disks.Sort((disk1, disk2)=> disk1[2].CompareTo(disk2[2]));
            int[] heights = new int[disks.Count];
            for(int i=0; i< disks.Count;i++){
                heights[i]=disks[i][2];
            }
            int[] sequences = new int[disks.Count];
            for(int i=0; i< disks.Count;i++){
                sequences[i]= Int32.MinValue;
            }

            int maxHeightIdx =0;
            for(int i=1; i< disks.Count; i++){
                int[] currentDisk = disks[i];
                for(int j=0; j<i; j++){
                    int[] otherDisk = disks[j];
                    if(areValidDimentions(otherDisk, currentDisk)){
                        if(heights[i] <= currentDisk[2]+ heights[j]){
                            heights[i] = currentDisk[2]+heights[j];
                            sequences[i] =j;
                        }
                    }
                }
                if(heights[i] >= heights[maxHeightIdx])                
                    maxHeightIdx = heights[i];
            }
            return BuildSequence(disks, sequences, maxHeightIdx);
            
        }

        private static List<int[]> BuildSequence(List<int[]> disks, int[] sequences, int currentIdx)
        {
            List<int[]> seq = new List<int[]>();
            while(currentIdx  != Int32.MinValue){
                seq.Insert(0, disks[currentIdx]);
                currentIdx = sequences[currentIdx];
            }
            return seq;
        }

        private static bool areValidDimentions(int[] otherDisk, int[] currentDisk)
        {
            return otherDisk[0] < currentDisk[0] && otherDisk[1] < currentDisk[1] && otherDisk[2] < currentDisk[2];
        }
        //https://www.algoexpert.io/questions/kruskals-algorithm
        //Minimum Spanning Tree(MST)
        public static int[][][] KrusaklMST(int[][][] edges) {
            
            //T:O(e*log(e)) | S: O(e+v)

            List<List<int>> sortedEdges = new List<List<int>>();
            for(int sourceIndex=0; sourceIndex < edges.Length; sourceIndex++){
                foreach(var edge in edges[sourceIndex]){
                    //skip reverse edges as this is an undirected graph
                    if(edge[0] > sourceIndex)
                    {
                        sortedEdges.Add(new List<int>(){sourceIndex, edge[0], edge[1]});                        
                    }
                }
            }
            sortedEdges.Sort((edge1, edge2) => edge1[2]-edge2[2]);
            int[] parents = new int[edges.Length];
            int[] ranks = new int[edges.Length];
            List<List<int[]>> mst = new List<List<int[]>>();

            for(int i=0; i< edges.Length; i++){
                parents[i] = i;
                ranks[i] =0;
                mst.Insert(i, new List<int[]>());
            }

            foreach(var edge in sortedEdges){
                int vertex1Root = Find(edge[0], parents);
                int vertex2Root = Find(edge[1], parents);

                if(vertex1Root != vertex2Root){
                    mst[edge[0]].Add(new int[]{edge[1], edge[2]});
                    mst[edge[1]].Add(new int[]{edge[0], edge[2]});
                    
                    Union(vertex1Root, vertex2Root, parents, ranks);
                }
      
            }

            int[][][] arrayMst =new int[edges.Length][][];
            for(int i=0; i< mst.Count; i++){
                arrayMst[i] = new int[mst[i].Count][];
                for(int j=0; j< mst[i].Count; j++){
                    arrayMst[i][j] = mst[i][j];
                }
            }
            return arrayMst;
        }

        private static void Union(int vertex1Root, int vertex2Root, int[] parents, int[] ranks)
        {
            if(ranks[vertex1Root] < ranks[vertex2Root]){
                parents[vertex1Root] = vertex2Root;
            }else if(ranks[vertex1Root] > ranks[vertex2Root]){
                parents[vertex2Root] = vertex1Root;
            }else {
                parents[vertex2Root] = vertex1Root;
                ranks[vertex1Root]++;
            }
        }

        private static int Find(int vertex, int[] parents)
        {
            if(vertex != parents[vertex]){
                parents[vertex] = Find(parents[vertex], parents); //Path Compression
            }
            return parents[vertex];
        }
        //https://www.algoexpert.io/questions/shorten-path
        //Ex: /foo/../test/../test/../foo//bar/./baz =>  /foo/bar/baz
        public static string ShortenPath(string path){
            //T:O(n)| S:O(n) - n is lenght of path
            bool startsWithPath =path[0] =='/';
            string[] tokensArr = path.Split("/");
            List<string> tokenList = new List<string>(tokensArr);
            List<string> filteredTokens = tokenList.FindAll(token => IsImportantToken(token));
            Stack<string> stack = new Stack<string>();
            if(startsWithPath) stack.Push("");
            foreach(string token in filteredTokens){
                if(token.Equals("..")){
                    if(stack.Count == 0 || stack.Peek().Equals("..")){
                        stack.Push(token);
                    }else if(!stack.Peek().Equals("")){
                        stack.Pop();
                    }
                }else{
                    stack.Push(token);
                }
              
            }
            if(stack.Count ==1 && stack.Peek().Equals("")) return "/";
            var arr= stack.ToArray();
            Array.Reverse(arr);
            return string.Join("/",arr);

        }

        private static bool IsImportantToken(string token)
        {
            return token.Length >0 && !token.Equals(".");
        }
        //https://www.algoexpert.io/questions/sweet-and-savory
        public static int[] SweetAndSavory(int[] dishes, int target){
            
            //T:O(nlog(n)) | S:O(n)
            List<int>  sweetDishes = new List<int>();
            List<int> savoryDishes = new List<int>();

            foreach(var dish in dishes){
                if(dish < 0)
                    sweetDishes.Add(dish);
                else 
                    savoryDishes.Add(dish);
                
            }
            sweetDishes.Sort((a,b)=> Math.Abs(a)-Math.Abs(b));
            savoryDishes.Sort();

            int[] bestPair = new int[2];
            int bestDiff= Int32.MinValue;
            int sweetIndx=0, savoryIndex=0;

            while(sweetIndx < sweetDishes.Count && savoryIndex < savoryDishes.Count){
                int currentSum = sweetDishes[sweetIndx] + savoryDishes[savoryIndex];
                
                if(currentSum <= target){
                    int currentDiff = target-currentSum;
                    if(currentDiff < bestDiff){
                        bestDiff = currentDiff;
                        bestPair[0] = sweetDishes[sweetIndx];
                        bestPair[1] = savoryDishes[savoryIndex];
                    }
                    savoryIndex++;
                }else sweetIndx++;
            }
            return bestPair;

        }
        //https://www.algoexpert.io/questions/stable-internships
        //Gale–Shapley algorithm or Stable match or Stable marriage algo
        public int[][] StableInternships(int[][] interns, int[][] teams){
            //T:O(n^2) | S:O(n^2)
            Dictionary<int, int> choosenInterns = new Dictionary<int, int>();
            Stack<int> freeInterns = new Stack<int>();
            for(int i=0; i< interns.Length; i++)
                freeInterns.Push(i);
            
            int[] currentInternChoices = new int[interns.Length];

            List<Dictionary<int, int>> teamDict = new List<Dictionary<int, int>>();
            foreach(var team in teams){
                Dictionary<int, int> rank = new Dictionary<int, int>();
                for(int i=0; i< team.Length; i++){
                    rank[team[i]]=i;
                }
                teamDict.Add(rank);                
            }
            while(freeInterns.Count >0){

                int internNum = freeInterns.Pop();

                int[] intern = interns[internNum];
                int teamPref = intern[currentInternChoices[internNum]];
                currentInternChoices[internNum]++;

                if(!choosenInterns.ContainsKey(teamPref)){
                    choosenInterns[teamPref] =internNum;
                    continue;
                }

                int prevIntern = choosenInterns[teamPref];
                int prevInternRank = teamDict[teamPref][prevIntern];
                int currentInternRank = teamDict[teamPref][internNum];

                if(currentInternRank < prevInternRank){
                    freeInterns.Push(prevIntern);
                    choosenInterns[teamPref] = internNum;
                }else {
                    freeInterns.Push(internNum);
                }
            }
            int[][] matches = new int[interns.Length][];
            int index=0;
            foreach(var choosenIntern in choosenInterns){
                matches[index] = new int[]{choosenIntern.Value, choosenIntern.Key};
                index++;
            }
            return matches;
        }

        //https://www.algoexpert.io/questions/group-anagrams
        public static List<List<string>> GroupAnagrams(List<string> words){
            
            if(words.Count ==0 ) return new List<List<string>>();
            //1. Naive n Complex
            //T:O(w*n*log(n)+n*w*log(w)) | S:O(wn) where w is the number of words and n is length of longest words
            List<List<string>> groupedAnagrams = GroupAnagramsNaive(words);

            //2. Optimal and Simple
            //T:O(w*n*log(n)) | S:O(wn) where w is the number of words and n is length of longest words            
            groupedAnagrams = GroupAnagramsOptimal(words);
            return groupedAnagrams;
        }

        private static List<List<string>> GroupAnagramsOptimal(List<string> words)
        {
            Dictionary<string, List<string>> anagrams = new Dictionary<string, List<string>>();

            foreach(string word in words){
                char[] chArr = word.ToCharArray();
                Array.Sort(chArr);
                string sorteWord = new string(chArr);
                if(anagrams.ContainsKey(sorteWord)){
                    anagrams[sorteWord].Add(word);
                }else{
                    anagrams[sorteWord] = new List<string>(){word};
                }
            }
            
            return anagrams.Values.ToList();
        }

        private static List<List<string>> GroupAnagramsNaive(List<string> words)
        {
            List<string> sortedWords = new List<string>();
            foreach(string word in words){
                char[] chArr = word.ToCharArray();
                Array.Sort(chArr);
                string sortedWord = new string(chArr);
                sortedWords.Add(sortedWord);
            }

            List<int> indices = Enumerable.Range(0, words.Count).ToList();
            indices.Sort((a,b) => sortedWords[a].CompareTo(sortedWords[b]));

            List<List<string>> result = new List<List<string>>();
            List<string> currentAnagramGrp = new List<string>();
            string currentAnagram = sortedWords[indices[0]];
            foreach(int index in indices){
                string word = words[index];
                string sortedWord = sortedWords[index];

                if(sortedWord.Equals(currentAnagram)){
                    currentAnagramGrp.Add(word);
                    continue;
                }

                result.Add(currentAnagramGrp);
                currentAnagramGrp = new List<string>();
                currentAnagram = sortedWord;
            }
            result.Add(currentAnagramGrp);
            return result;
        }
        //https://www.algoexpert.io/questions/valid-ip-addresses
        public static List<string> ValidIPAddresses(string str){

            //T:O(1) as in 2^32 IP addresses for full 12 digit | S:O(1)
        
            List<string> ipAddresessFound  = new List<string>();

            for(int i=1; i< Math.Min((int)str.Length, 4); i++)
            {
                string[] currentIPAddressParts = new string[]{"","","",""};

                currentIPAddressParts[0] = str.Substring(0, i-0);
                if(!IsValidPart(currentIPAddressParts[0])){
                    continue;
                }
                for(int j=i+1; j< i+Math.Min((int)str.Length-i,4); j++){ //or j< Math.Min((int)str.Length, i+4)
                    currentIPAddressParts[1] = str.Substring(i, j-i);
                    if(!IsValidPart(currentIPAddressParts[1]))
                        continue;
                    
                    for(int k=j+1; k< j+Math.Min((int)str.Length-j,4); k++){
                        currentIPAddressParts[2] = str.Substring(j,k-j);
                        currentIPAddressParts[3] = str.Substring(k);

                        if(IsValidPart(currentIPAddressParts[2]) && IsValidPart(currentIPAddressParts[3])){
                            ipAddresessFound.Add(JoinParts(currentIPAddressParts));
                        }
                    }
                }

                
            }
            return ipAddresessFound;

        }

        private static string JoinParts(string[] currentIPAddressParts)
        {
            StringBuilder sb = new StringBuilder();
            for(int i=0; i< currentIPAddressParts.Length; i++){
                sb.Append(currentIPAddressParts[i]);
                if(i< currentIPAddressParts.Length)
                    sb.Append(".");
            }
            return sb.ToString();
        }

        private static bool IsValidPart(string str)
        {
            int stringAsInt = Int32.Parse(str);
            if(stringAsInt > 255) return false;
            return str.Length == stringAsInt.ToString().Length; //Check for leading 0's
        }
        
        //https://www.algoexpert.io/questions/staircase-traversal
        public static int StaircaseTraversal(int height, int maxSteps){

            
            //1.Naive - Recursion
            //T:O(k^n) - exponetial | S:O(n) where n is height of staircase and k is number of allowed steps
            int numberOfWaysToTop = NumberOfWaysToTopNaiveRec(height, maxSteps)  ;

            //2.Optimal - Recursion with Memorization (DP)
            //T:O(k*n) | S:O(n) where n is height of staircase and k is number of allowed steps
            Dictionary<int, int> memoize = new Dictionary<int, int>();
            memoize[0]=1;
            memoize[1]=1;
            numberOfWaysToTop = NumberOfWaysToTopOptimalRec(height, maxSteps, memoize)  ;

            //3.Optimal - Iterative with Memorization (DP)
            //T:O(k*n) | S:O(n) where n is height of staircase and k is number of allowed steps
            numberOfWaysToTop = NumberOfWaysToTopOptimalIterative(height, maxSteps)  ;

            //4.Optimal - Iterative with Memorization (DP)  
            //T:O(n) | S:O(n) where n is height of staircase and k is number of allowed steps
            numberOfWaysToTop = NumberOfWaysToTopOptimalIterative2(height, maxSteps)  ;
            
            return numberOfWaysToTop;
        }

        private static int NumberOfWaysToTopOptimalIterative2(int height, int maxSteps)
        {
            int curNumOfWays = 0;
            List<int> waysToTop=  new List<int>();
            waysToTop.Add(1);
            for(int curHeight=1; curHeight < height+1; curHeight++){

                int startOfWindow =curHeight-maxSteps-1;
                int endOfWindow = curHeight-1;

                if(startOfWindow >=0){
                    curNumOfWays -=waysToTop[startOfWindow];
                }
                curNumOfWays += waysToTop[endOfWindow];
                waysToTop.Add(curNumOfWays);
            }
            return waysToTop[height];

        }

        private static int NumberOfWaysToTopOptimalIterative(int height, int maxSteps)
        {
            int[] waysToTop = new int[height+1];
            waysToTop[0]=1;
            waysToTop[1]=1;
            for(int curHeight=2; curHeight < height+1; curHeight++){
                int step=1;
                while(step <= maxSteps && step<=curHeight){
                    waysToTop[curHeight] = waysToTop[curHeight]+waysToTop[curHeight-step];
                    step++;
                }

            }
            return waysToTop[height];

        }

        private static int NumberOfWaysToTopOptimalRec(int height, int maxSteps, Dictionary<int, int> memoize)
        {
            if(memoize.ContainsKey(height))
                return memoize[height];
            
            int numberOfWays =0;

            for(int step=1; step< Math.Min(maxSteps, height)+1; step++)
            {
                numberOfWays += NumberOfWaysToTopOptimalRec(height-step, maxSteps,memoize);
            }
            memoize[height] = numberOfWays;
            
            return numberOfWays;
            
        }

        private static int NumberOfWaysToTopNaiveRec(int height, int maxSteps)
        {
            if(height <=1)
                return 1;
            int numberOfWays = 0;
            for(int step=1; step < Math.Min(maxSteps, height)+1; step++){
                numberOfWays += NumberOfWaysToTopNaiveRec(height-step, maxSteps);
            }
            return numberOfWays;
        }
        //https://www.algoexpert.io/questions/reversePolishNotation
        public static int ReversePolishNotation(string[] tokens) {
            Stack<int> operands = new Stack<int>();
            foreach(string token in tokens){
                if(token.Equals("+")){
                operands.Push(operands.Pop() + operands.Pop());        
                }else if(token.Equals("-")){
                int firstNum = operands.Pop();
                operands.Push(operands.Pop() - firstNum);   
                }else if(token.Equals("*")){
                operands.Push(operands.Pop() * operands.Pop());
                }else if(token.Equals("/")){
                int firstNum = operands.Pop();
                operands.Push(operands.Pop()/firstNum);        
                }else {
                    operands.Push(Int32.Parse(token));        
                }
            }
            return operands.Pop();
         }
    
        //https://www.algoexpert.io/questions/evaluate-expression-tree
        public int EvaluateExpressionTree(BinaryTree tree) {
            //T:O(n) time | S: O(h) space - where n is the number of nodes in the Binary Tree, and h is the height of the Binary Tree
            return EvalExpTree(tree);        
        }
        private int EvalExpTree(BinaryTree node){

            if(node.Value >=0 ) return node.Value;
            
            int left= EvalExpTree(node.Left);
            int right = EvalExpTree(node.Right);
            int res = EvalExp(left, right, node.Value);

            return res;
            
          }
        private int EvalExp(int op1, int op2, int opera ){
            if(opera == -1)
                return op1+op2;
            else if(opera == -2) 
                return op1-op2;
            else if(opera == -3)
                return op1/op2;
            else if (opera == -4)
                return op1*op2;
            else 
                return opera;
        }

        //https://www.algoexpert.io/questions/water-area
        //rain water trapped
        public static int WaterArea(int[] heights){

            if(heights.Length == 0) return 0;
            //1. Using auxiliary space
            //T:O(n) | S:O(n)
            int trappedWater = WaterArea1(heights);
            
            //2. Without Using auxiliary space
            //T:O(n) | S:O(1)
            trappedWater = WaterArea2(heights);

            return trappedWater;
        }

        private static int WaterArea2(int[] heights)
        {
            int leftIdx =0, rightIdx =heights.Length-1;
            int leftMax = heights[leftIdx], rightMax = heights[rightIdx];
            var trappedWater=0;

            while(leftIdx < rightIdx){
                if(heights[leftIdx] < heights[rightIdx]){
                    leftIdx++;
                    leftMax= Math.Max(leftMax, heights[leftIdx]);
                    trappedWater +=leftMax - heights[leftIdx];
                }else{
                    rightIdx--;
                    rightMax = Math.Max(rightMax, heights[rightIdx]);
                    trappedWater += rightMax - heights[rightIdx];
                }
            }
            return trappedWater;
        }

        private static int WaterArea1(int[] heights)
        {
            int[] maxes = new int[heights.Length];
            int leftMax =0;
            for(int i=0; i< heights.Length; i++){
                int height = heights[i];
                maxes[i] =leftMax;
                leftMax = Math.Max(leftMax, height);
            }

            int rightMax =0;
            for(int i=heights.Length-1; i>=0; i--){
                int height = heights[i];
                int minHeight = Math.Min(maxes[i], rightMax);
                if(height < minHeight){
                    maxes[i] = minHeight-height;                    
                }else maxes[i]=0;
                
                rightMax = Math.Max(rightMax, height);

            }
            int total =0;
            for(int idx=0; idx<maxes.Length; idx++){
                total += maxes[idx];
            }
            return total;
        }
        //https://www.algoexpert.io/questions/optimal-freelancing
        public static int OptimalFreelancing(Dictionary<string , int>[] jobs){
            //1.Naive - with pair of loops to compare max profit against each combination
            //T:O(n^2) : SO(1)

            //2.T:O(nlogn) | S:O(1)
            const int LENGTH_OF_PERIOD = 7;

            int maxProfit =0;

               
            Array.Sort(
                    jobs, 
                    Comparer<Dictionary<string, int>>.Create(
                        (jobOne, jobTwo)=> jobTwo["payment"].CompareTo(jobOne["payment"])
                        )
                    );

            bool[] timeLine = new bool[LENGTH_OF_PERIOD];
            foreach(var job in jobs){
                int maxTime = Math.Min(job["deadline"], LENGTH_OF_PERIOD);
                for(int time = maxTime-1; time>=0; time--){
                    if(!timeLine[time]){
                        timeLine[time] = true;
                        maxProfit += job["payment"];
                        break;
                    }
                }
            }
            return maxProfit;

        }
        //https://www.algoexpert.io/questions/prims-algorithm
        //Minimum Spanning Tree (MST)

        public static int[][][] PrimsMST(int[][][] edges){
            //T:O(e*log(v)) | S:O(v+e) - e is number of edges and v is number of vertices.
            List<Item> initialEdgeItems = new List<Item>();
            foreach(var edge in edges[0]){
                Item edgeItem = new Item(0, edge[0], edge[1]);
                initialEdgeItems.Add(edgeItem);
            }
            Heap<Item> minHeap = new Heap<Item>(initialEdgeItems, CompareByDistance);
            
            List<List<int[]>> mst = new List<List<int[]>>();
            for(int i=0; i<edges.Length; i++){
                mst.Add(new List<int[]>());                
            }
            while(!minHeap.IsEmpty()){
                Item heapItem = minHeap.Remove();
                int vertex = heapItem.Vertex;
                int discoverdVertex = heapItem.DiscoverdVertex;
                int distance = heapItem.Distance;

                if(mst[discoverdVertex].Count == 0){
                    mst[vertex].Add(new int[]{discoverdVertex, distance});
                    mst[discoverdVertex].Add(new int[]{vertex, distance});
                
                    foreach(var edge in edges[discoverdVertex]){
                        int neighbor = edge[0];
                        int neighborDistance =edge[1];

                        if(mst[neighbor].Count ==0){
                            minHeap.Insert(new Item(discoverdVertex , neighbor, neighborDistance));
                        }
                    }
                }
            }
            int[][][] arrayMST = new int[edges.Length][][];
            for(int i=0; i< mst.Count; i++){
                arrayMST[i] = new int[mst[i].Count][];
                for(int j=0; j<mst[i].Count; j++){
                    arrayMST[i][j] = mst[i][j];
                }
            }
            return arrayMST;
        }
        private static bool CompareByDistance(Item item1, Item item2){

            return item1.Distance < item2.Distance;
        }
         class Item{
            public int Vertex;
            public int DiscoverdVertex;
            public int Distance;

            public Item(int vertex, int discoverdVertex, int distance){
                this.Vertex=vertex;
                this.DiscoverdVertex=discoverdVertex;
                this.Distance = distance;
            }
        }
        //https://www.algoexpert.io/questions/topological-sort
        public static List<int> TopologicalSort(List<int> jobs, List<int[]> deps){

            //1. Picking random node/job and then DFS
            //T:O(j+d) | S:O(j+d) - j and d are jobs and dependencies
            List<int> orderedJobs = TopologicalSort1(jobs, deps);

            //2. Picking non dependent nodes first and then DFS
            //T:O(j+d) | S:O(j+d) - j and d are jobs and dependencies
            orderedJobs = TopologicalSort2(jobs, deps);


            return orderedJobs;
        }

        private static List<int> TopologicalSort2(List<int> jobs, List<int[]> deps)
        {
              JobGraph jobGraph = CreateJobGraph(jobs, deps);
            return GetOrderedJobs(jobGraph);
        }

        private static List<int> TopologicalSort1(List<int> jobs, List<int[]> deps)
        {
            JobGraph jobGraph = CreateJobGraph(jobs, deps);
            return GetOrderedJobs1(jobGraph);
        }
        private static JobGraph CreateJobGraph1(List<int> jobs, List<int[]> deps)
        {
            JobGraph jobGraph  = new JobGraph(jobs);
            foreach(int[] dep in deps){
                jobGraph.AddDep(dep[0], dep[1]);
            }
            return jobGraph;
        }
        private static List<int> GetOrderedJobs1(JobGraph jobGraph)
        {
            List<int> orderedJobs = new List<int>();
            Stack<JobNode> nodesWithNoPrereqs = new Stack<JobNode>();
            foreach(JobNode node in jobGraph.nodes){
                if(node.NumOfPrereqs == 0) nodesWithNoPrereqs.Push(node);
            }
            while(nodesWithNoPrereqs.Count > 0){
                JobNode node = nodesWithNoPrereqs.Pop();
                orderedJobs.Add(node.Job);
                RemoveDeps(node, nodesWithNoPrereqs);
            }
            bool graphHasEdges = false; //Cycle check
            foreach(JobNode jobNode in jobGraph.nodes){
                if(jobNode.NumOfPrereqs >0) graphHasEdges = true;
            }
            return graphHasEdges ? new List<int>() : orderedJobs;
        }

        private static void RemoveDeps(JobNode node, Stack<JobNode> nodesWithNoPrereqs)
        {
            while(node.Deps.Count > 0){
                JobNode dep = node.Deps[node.Deps.Count-1];
                node.Deps.RemoveAt(node.Deps.Count-1);
                dep.NumOfPrereqs--;
                if(dep.NumOfPrereqs == 0) nodesWithNoPrereqs.Push(dep);
            }
        }

        private static List<int> GetOrderedJobs(JobGraph jobGraph)
        {
            List<int> orderedJobs = new List<int>();
            Stack<JobNode> nodeStack = jobGraph.nodes; 
            while(nodeStack.Count > 0 ){
                JobNode node = nodeStack.Pop();
                bool containsCycle = DepthFirstTraverse(node, orderedJobs);
                if(containsCycle) return new List<int>();

            }
            return orderedJobs;
        }

        private static bool DepthFirstTraverse(JobNode node, List<int> orderedJobs)
        {
            if(node.Visited) return false;
            if(node.Visiting) return true;
            node.Visiting = true;
            foreach(JobNode prereqNode in node.Prereqs){
                bool containsCycle = DepthFirstTraverse(prereqNode, orderedJobs);
                if(containsCycle) return true;
            }
            node.Visited = true;
            node.Visiting = false;
            orderedJobs.Add(node.Job);
            return false;
        }

        private static JobGraph CreateJobGraph(List<int> jobs, List<int[]> deps)
        {
            JobGraph jobGraph  = new JobGraph(jobs);
            foreach(int[] dep in deps){
                jobGraph.AddPrereq(dep[1], dep[0]);
            }
            return jobGraph;
        }
        public class JobGraph{
            public Stack<JobNode> nodes;
            public Dictionary<int, JobNode> graph;

            public JobGraph(List<int> jobs){
                nodes = new Stack<JobNode>();
                graph = new Dictionary<int, JobNode>();
                foreach(int job in jobs){
                    AddNode(job);
                }
            }
           private void AddNode(int job){
                graph.Add(job, new JobNode(job));
                nodes.Push(graph[job]);

           }
           public void AddPrereq(int job, int prereq){ 
            JobNode jobNode = GetNode(job);
            JobNode prereqNode = GetNode(prereq);
            jobNode.Prereqs.Add(prereqNode);                        

           }

            public void AddDep(int job, int dep){  //For apporach#2
            JobNode jobNode = GetNode(job);
            JobNode depNode = GetNode(dep);
            jobNode.Deps.Add(depNode);  
            depNode.NumOfPrereqs++;                      

           }

            private JobNode GetNode(int job)
            {
                if(!graph.ContainsKey(job)) AddNode(job);
                return graph[job];
            }
        }
        public class JobNode{
            public int Job;
            public List<JobNode> Prereqs; 
            public List<JobNode> Deps;  //for Approach 2
            public bool Visited;
            public bool Visiting;
            public int NumOfPrereqs;
            public JobNode(int job){
                this.Job = job;
                this.Prereqs = new List<JobNode>();
                Visited = false;
                Visiting = false;
                NumOfPrereqs=0;
            }

        }
        //https://www.algoexpert.io/questions/juice-bottling
        public static List<int> JuiceBottling(int[] prices){
            
            TODO:
            //1.bruteforce 
            //Find all possible pairs and sum to check for MaxProfit

            //2.Naive
            //T:O(n^3) | S:O(n^2) - n is length of prices
            List<int> maxProfit = JuiceBottlingNaive(prices);

            //3.Optimal
            //T:O(n^2) | S:O(n) - n is length of prices
            maxProfit = JuiceBottlingOptimal(prices);

            return maxProfit;
        }

        private static List<int> JuiceBottlingOptimal(int[] prices)
        {
            int numSizes = prices.Length;
            int[] maxProfit = new int[numSizes];
            int[] dividingPoints = new int[numSizes];

            for(int size=0; size< numSizes; size++){
                for(int dividingPoint=0; dividingPoint <= size; dividingPoint++){
                    int possibleProfit = maxProfit[size-dividingPoint]+prices[dividingPoint];

                    if(possibleProfit > maxProfit[size]){
                        maxProfit[size] = possibleProfit;
                        dividingPoints[size] = dividingPoint;
                    }
                }
            }
            List<int> solution = new List<int>();
            int curDividingPoint = numSizes-1;
            while(curDividingPoint > 0){
                solution.Add(dividingPoints[curDividingPoint]);
                curDividingPoint -= dividingPoints[curDividingPoint];
            }
            return solution;
        }

        private static List<int> JuiceBottlingNaive(int[] prices)
        {
            int numSizes = prices.Length;
            int[] maxProfilt = new int[numSizes];
            List<List<int>> solutions = new List<List<int>>();
            for(int size =0; size < numSizes; size++){
                solutions.Add(new List<int>());
            }
            for(int size=0; size< numSizes;size++){ //O(n)
                for(int dividingPoint =0; dividingPoint <=size; dividingPoint++){ //O(n)
                    int possibleProfit = maxProfilt[size-dividingPoint]+prices[dividingPoint];

                    if(possibleProfit > maxProfilt[size]){
                        maxProfilt[size]= possibleProfit;
                        List<int>  newSolution = new List<int>(); 
                        newSolution.Add(dividingPoint);
                        newSolution.AddRange(solutions[size-dividingPoint]); ////O(n)
                        solutions[size] = newSolution;
                    }
                }
            }
            return solutions[numSizes-1];
        }

        //https://www.algoexpert.io/questions/detect-arbitrage
        public static bool DetectArbitrage(List<List<Double>> exchangeRates){

            TODO:
            // O(n^3) time | O(n^2) space - where n is the number of currencies
            
            // To use exchange rates as edge weights, we must be able to add them.
            // Since log(a*b) = log(a) + log(b), we can convert all rates to
            // -log10(rate) to use them as edge weights.
            List<List<Double> > logExchangeRates = ConvertToLogMatrix(exchangeRates);

            // A negative weight cycle indicates an arbitrage.
            return FoundNegativeWeightCycle(logExchangeRates, 0);
        }

         // Runs the Bellman–Ford Algorithm to detect any negative-weight cycles.
        private static bool FoundNegativeWeightCycle(List<List<double>> graph, int start)
        {
            double[] distancesFromStart = new double[graph.Count];
            Array.Fill(distancesFromStart, Double.MaxValue);
            distancesFromStart[start] = 0;

            for (int unused = 0; unused < graph.Count; unused++) {
            // If no update occurs, that means there's no negative cycle.
            if (!RelaxEdgesAndUpdateDistances(graph, distancesFromStart)) {
                return false;
            }
            }

            return RelaxEdgesAndUpdateDistances(graph, distancesFromStart);
        }

        private static List<List<double>> ConvertToLogMatrix(List<List<double>> matrix)
        {
            List<List<Double> > newMatrix = new List<List<Double> >();

            for (int row = 0; row < matrix.Count; row++) {
            List<Double> rates = matrix[row];
            newMatrix.Add(new List<Double>());
            foreach (var rate in rates) {
                newMatrix[row].Add(-Math.Log10(rate));
            }
            }
        return newMatrix;
        }
       
       

        private static bool RelaxEdgesAndUpdateDistances(List<List<double>> graph, double[] distances)
        {
            bool updated = false;

            for (int sourceIdx = 0; sourceIdx < graph.Count; sourceIdx++) {
            List<Double> edges = graph[sourceIdx];
            for (int destinationIdx = 0; destinationIdx < edges.Count;
                destinationIdx++) {
                double edgeWeight = edges[destinationIdx];
                double newDistanceToDestination = distances[sourceIdx] + edgeWeight;
                if (newDistanceToDestination < distances[destinationIdx]) {
                updated = true;
                distances[destinationIdx] = newDistanceToDestination;
                }
            }
            }

            return updated;    
        }
        //https://www.algoexpert.io/questions/task-assignment
        public static List<List<int>> TaskAssignment(int k, List<int> tasks){
            //1.Naive - Generate all possible pairs and check for optimal duration
            //T:O(n^2)  | S:O(1)

            //2.Sorting to join largest duration with smallest duration inorder to achieve optimal duration 
            //T:O(nlogn) | S:O(n)
            List<List<int>> pairedTasks = new List<List<int>>();
            Dictionary<int, List<int>> taskDurationsToIndices = GetTaskDurationsToIndices(tasks);
            
            tasks.Sort();

            for(int idx=0; idx< k; idx++){

                int task1Duration = tasks[0]; //Sorted one
                
                List<int> indicesWithTask1Duration = taskDurationsToIndices[task1Duration];
                int task1OriginalIndex = indicesWithTask1Duration[indicesWithTask1Duration.Count-1];
                indicesWithTask1Duration.RemoveAt(indicesWithTask1Duration.Count-1);

                int task2Index = tasks.Count-1-idx; //Sorted index
                int task2Duration = tasks[task2Index];
                List<int> indicesWithTask2Duration = taskDurationsToIndices[task2Duration];
                int task2OriginalIndex = indicesWithTask2Duration[indicesWithTask2Duration.Count-1];
                indicesWithTask2Duration.RemoveAt(indicesWithTask2Duration.Count-1);

                List<int> pairedTask = new List<int>
                {
                    task1OriginalIndex,
                    task2OriginalIndex
                };
                pairedTasks.Add(pairedTask);

            }
            return pairedTasks;
        }

        private static Dictionary<int, List<int>> GetTaskDurationsToIndices(List<int> tasks)
        {
            Dictionary<int, List<int>> taskDurationsToIndices = new Dictionary<int, List<int>>();

            for(int idx=0; idx< tasks.Count; idx++){
                int taskDuration = tasks[idx];
                if(taskDurationsToIndices.ContainsKey(taskDuration)){
                    taskDurationsToIndices[taskDuration].Add(idx);
                }else{
                    List<int> temp = new List<int>();
                    temp.Add(idx);
                    taskDurationsToIndices[taskDuration]= temp;
                }
            }

            return taskDurationsToIndices;
            
        }
        //https://www.algoexpert.io/questions/sunset-views
        public static List<int> SunsetViews(int[] buildings, string direction){

            //1.Naive - pair of loops to find whether each building can see sunset
            //T:O(n^2) | S:O(n)

            //2.Optimal - keep track of maxheight to not to loop thru again n again
            //T:O(n) | S:O(n)
            List<int> sunsetViews = SunsetViewsOptimal1(buildings, direction);

            //3.Optimal - using Stack to keep track of maxheight to not to loop thru again n again
            //T:O(n) | S:O(n)
            sunsetViews = SunsetViewsOptimal2(buildings, direction);

            return sunsetViews;
        }

        private static List<int> SunsetViewsOptimal2(int[] buildings, string direction)
        {
            Stack<int> candidateBuildings = new Stack<int>();

            
            int startIdx = buildings.Length-1;
            int step =-1;

            if(direction.Equals("EAST")){
                startIdx=0;
                step =1;
            }

            int idx=startIdx;
            while(idx>=0 && idx < buildings.Length){
                int buildingHeight = buildings[idx];

                while(candidateBuildings.Count >0 && candidateBuildings.Peek() <= buildingHeight){
                    candidateBuildings.Pop();
                }
                candidateBuildings.Push(idx);

                idx +=step;
            }
            if(direction.Equals("WEST")) candidateBuildings.Reverse();
            
            return candidateBuildings.ToList();

        }

        private static List<int> SunsetViewsOptimal1(int[] buildings, string direction)
        {
            List<int> buildingsWithSunsetViews = new List<int>();

            int startIdx = buildings.Length-1;
            int step =-1;

            if(direction.Equals("EAST")){
                startIdx=0;
                step =1;
            }

            int idx=startIdx;
            int runningMaxHeight =0;
            while(idx>=0 && idx< buildings.Length){
                int buildingHeight = buildings[idx];

                if(buildingHeight > runningMaxHeight){
                    buildingsWithSunsetViews.Add(idx);
                }
                runningMaxHeight = Math.Max(runningMaxHeight, buildingHeight);

                idx += step;

            }
            if(direction.Equals("WEST")) buildingsWithSunsetViews.Reverse();

            return buildingsWithSunsetViews;
        }
        //https://www.algoexpert.io/questions/ambiguous-measurements
        public static bool AmbiguousMeasurements(int[][] measuringCups, int low, int high){
            //DP Recursion with memoization
            //T:O(low*high*n) | S:O(low*high)
            Dictionary<string, bool> memoization = new Dictionary<string, bool>();
            return CanMeasureInRange(measuringCups, low, high, memoization);
        }

        private static bool CanMeasureInRange(int[][] measuringCups, int low, int high, Dictionary<string, bool> memoization)
        {
            string memoizeKey = CreateHashTableKey(low, high);

            if(memoization.ContainsKey(memoizeKey))
                return memoization[memoizeKey];
            
            if(low <=0 && high <=0){
                return false;
            }
            bool canMeasure = false;
            foreach(var cup in measuringCups){
                int cupLow = cup[0];
                int cupHigh = cup[1];
                if(low <= cupLow && cupHigh <=high){
                    canMeasure = true;
                    break;
                }
                int newLow = Math.Max(0, low-cupLow);
                int newHigh = Math.Max(0, high-cupHigh);
                canMeasure = CanMeasureInRange(measuringCups, newLow, newHigh, memoization);
                if(canMeasure) break;
            }
            memoization[memoizeKey]= canMeasure;
            return canMeasure;
        }

        private static string CreateHashTableKey(int low, int high)
        {
            return low.ToString()+"-"+ high.ToString();
        }
        //https://www.algoexpert.io/questions/largest-park
        //Largest Rectangle in Histogram
        public static int LargestPark(bool[][] land){
            //T:O(w*h) | S:O(w) - w and h are width(row) and height(column) of input matrix
            int[] heights = new int[land[0].Length];
            int maxArea =0;
            foreach(var row in land){
                for(int col=0; col < land[0].Length; col++){
                    heights[col] = row[col] == false? heights[col]+1:0;
                }
                maxArea = Math.Max(maxArea, LargestRectangleInHistogram(heights));
            }
            return maxArea;

        }

        private static int LargestRectangleInHistogram(int[] heights)
        {
            Stack<int> stack = new Stack<int>();
            int maxArea =0;
            for(int col=0; col< heights.Length; col++){
                while(stack.Count > 0 && heights[col] < heights[stack.Peek()]){
                    int height = heights[stack.Pop()];
                    int width = (stack.Count ==0) ? col : col-stack.Peek()-1;

                    maxArea = Math.Max(maxArea, width* height);

                }
                stack.Push(col);
            }
            //For remain elements
            while(stack.Count>0){
                int height = heights[stack.Pop()];
                int width = (stack.Count==0)?heights.Length: heights.Length-stack.Peek()-1;
                maxArea = Math.Max(maxArea, height*width);
            }
            return maxArea;
        }
        //https://www.algoexpert.io/questions/laptop-rentals
        public static int LaptopRentals(List<List<int>> times){
            //1.Naive - Pair of loops to find laptop reuse via non-overlapping intervals
            //T:O(n^2) | S:O(1)

            //2.Naive - Sorting by starttime but still need loop thru previusly visited times to find a free laptop
            //T:O(n^2) | S:O(1)

            //3.Optimal- Using MinHeap along sorting by startime to find overlapping intervals via end time of previous and start time of current intervals
            //T:O(nlogn) | S:O(n)
            int minLaptops = LaptopRentalsOptimal1(times);

            //4.Optimal- Disecting start and end times then use Two pointers with Sorting by startime to find overlapping intervals via end time of previous and start time of current intervals
            //T:O(nlogn) | S:O(n)
            minLaptops = LaptopRentalsOptimal2(times);

            return minLaptops;
        }

        private static int LaptopRentalsOptimal2(List<List<int>> times)
        {
             if(times.Count ==0 ) return 0;
             
             int usedLaptops =0;
             List<int> startTimes = new List<int>();
             List<int> endTimes = new List<int>();

             foreach(var interval in times){
                startTimes.Add(interval[0]);
                endTimes.Add(interval[1]);
             }

             startTimes.Sort();
             endTimes.Sort();

             int startIterator =0;
             int endIterator = 0;

             while(startIterator < times.Count){
                if(startTimes[startIterator] >= endTimes [endIterator]){ // If no overalp then reducing laptop count to indicate laptop reuse
                    usedLaptops -=1;
                    endIterator +=1;
                }
                usedLaptops +=1;
                startIterator +=1;
             }
             return usedLaptops;
           
        }

        private static int LaptopRentalsOptimal1(List<List<int>>  times)
        {
            if(times.Count ==0 ) return 0;
            times.Sort((a,b)=>a[0.CompareTo(b[0])]);

            List<List<int>> timesWhenLaptopIsUsed = new List<List<int>>();
            timesWhenLaptopIsUsed.Add(times[0]);
            Heap<List<int>> minHeap = new Heap<List<int>>(timesWhenLaptopIsUsed, (a,b)=>{ return a[0]<b[0];});

            for(int idx=1; idx < times.Count;idx++){
                List<int> currentInterval = times[idx];
                if(minHeap.Peek()[1] <= currentInterval[0]){ // If no overalp then removing a time to indicate laptop reuse
                    minHeap.Remove();
                }
                minHeap.Insert(currentInterval);
            }
            return timesWhenLaptopIsUsed.Count; //or minHeap.Count;

        }
    }
    


    
}