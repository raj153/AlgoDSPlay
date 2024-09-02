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
    public partial class RealProbs
    {


        //https://www.algoexpert.io/questions/run-length-encoding
        public static string RunLengthEncoding(string str)
        {
            //T:O(n)|S:O(n)
            StringBuilder encodedStringChars = new StringBuilder();
            int currentRunLength = 1;

            for (int i = 1; i < str.Length; i++)
            {
                char currentChar = str[i];
                char previousChar = str[i - 1];

                if ((currentChar != previousChar) || (currentRunLength == 9))
                {
                    encodedStringChars.Append(currentRunLength.ToString());
                    encodedStringChars.Append(previousChar);
                    currentRunLength = 0;
                }
                currentRunLength += 1;
            }

            return encodedStringChars.ToString();
        }
        //https://www.algoexpert.io/questions/tandem-bicycle
        public int TandemCycle(int[] redShirtSpeeds, int[] blueShirtSpeeds, bool fastest)
        {
            //T: O(nlog(n))| S:O(1)
            Array.Sort(redShirtSpeeds);
            Array.Sort(blueShirtSpeeds);

            if (fastest)
                reverseArrayInPlace(redShirtSpeeds);

            int totalSpeed = 0;
            for (int idx = 0; idx < redShirtSpeeds.Length; idx++)
            {
                int rider1 = redShirtSpeeds[idx];
                int rider2 = blueShirtSpeeds[idx];
                totalSpeed += Math.Max(rider1, rider2);
            }
            return totalSpeed;

        }

        private void reverseArrayInPlace(int[] array)
        {
            int start = 0;
            int end = array.Length - 1;
            while (start < end)
            {
                int temp = array[start];
                array[start] = array[end];
                array[end] = temp;
                start++;
                end--;
            }
        }

        //https://www.algoexpert.io/questions/class-photos
        public static bool CanTakeClassPhotos(List<int> redShirtHeights, List<int> blueShirtHeights)
        {

            //T: O(Nlog(N)) | S: O(1)
            redShirtHeights.Sort((a, b) => b.CompareTo(a));
            blueShirtHeights.Sort((a, b) => b.CompareTo(a));

            string shirtColorInFirstRow = (redShirtHeights[0] < blueShirtHeights[0] ? "RED" : "BLUE");

            for (int idx = 0; idx < redShirtHeights.Count; idx++)
            {
                int redShirtHeight = redShirtHeights[idx];
                int blueShirtHeight = blueShirtHeights[idx];

                if (shirtColorInFirstRow == "RED")
                    if (redShirtHeight >= blueShirtHeight) return false;
                    else
                    {
                        if (blueShirtHeight >= redShirtHeight) return false;
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
            // O(c1 + c2) time | O(c1 + c2) space - where c1 and c2 are the respective
            // numbers of meetings in calendar1 and calendar2
            List<Meeting> updateCalendar1 = UpdateCalendar(calendar1, dailyBounds1);
            List<Meeting> updateCalendar2 = UpdateCalendar(calendar2, dailyBounds2);
            List<Meeting> mergedCalendar = MergeCalendars(updateCalendar1, updateCalendar2);
            List<Meeting> mergeIntervals = MergeIntervals(mergedCalendar);

            return GetMatchingAvailabilities(mergeIntervals, meetringDuration);
        }

        private static List<StringMeeting> GetMatchingAvailabilities(List<Meeting> mergeIntervals, int meetringDuration)
        {
            List<StringMeeting> availableMeetingSlots = new List<StringMeeting>();

            for (int i = 1; i < mergeIntervals.Count; i++)
            {
                int start = mergeIntervals[i - 1].End;
                int end = mergeIntervals[i].Start;
                int availabilityDuration = end - start;
                if (availabilityDuration >= meetringDuration)
                    availableMeetingSlots.Add(new StringMeeting(MinutesToTime(start), MinutesToTime(end)));

            }
            return availableMeetingSlots;
        }

        private static List<Meeting> MergeIntervals(List<Meeting> calendar)
        {
            List<Meeting> mergedIntervals = new List<Meeting>();
            mergedIntervals.Add(calendar[0]);
            for (int i = 1; i < calendar.Count; i++)
            {
                Meeting currentMeeting = calendar[i];
                Meeting previousMeeting = mergedIntervals[mergedIntervals.Count - 1];
                //6-11 8-10
                if (currentMeeting.Start <= previousMeeting.End)
                    mergedIntervals[mergedIntervals.Count - 1].End = Math.Max(currentMeeting.End, previousMeeting.End);
                else
                    mergedIntervals.Add(currentMeeting);
            }
            return mergedIntervals;
        }

        private static List<Meeting> MergeCalendars(List<Meeting> calendar1, List<Meeting> calendar2)
        {
            List<Meeting> merged = new List<Meeting>();

            int i = 0, j = 0;
            while (i < calendar1.Count && j < calendar2.Count)
            {
                Meeting meeting1 = calendar1[i];
                Meeting meeting2 = calendar2[j];

                if (meeting1.Start < meeting2.Start)
                {
                    merged.Add(meeting1);
                    i++;
                }
                else
                {
                    merged.Add(meeting2);
                    j++;
                }
            }
            while (i < calendar1.Count) merged.Add(calendar1[i]);
            while (j < calendar2.Count) merged.Add(calendar2[j]);

            return merged;

        }

        private static List<Meeting> UpdateCalendar(List<StringMeeting> calendar, StringMeeting dailyBounds)
        {
            List<StringMeeting> updatedCalendar = new List<StringMeeting>();
            updatedCalendar.Add(new StringMeeting("0:00", dailyBounds.Start));
            updatedCalendar.AddRange(calendar);
            updatedCalendar.Add(new StringMeeting(dailyBounds.End, "23:59"));
            List<Meeting> calendarInMins = new List<Meeting>();
            for (int i = 0; i < updatedCalendar.Count; i++)
            {
                calendarInMins.Add(new Meeting(TimeToMinuts(updatedCalendar[i].Start), TimeToMinuts(updatedCalendar[i].End)));
            }
            return calendarInMins;
        }
        private static int TimeToMinuts(string time)
        {
            string[] delimProps = time.Split(':');
            int hours = Int32.Parse(delimProps[0]);
            int minutes = Int32.Parse(delimProps[1]);
            return hours * 60 + minutes;


        }
        private static string MinutesToTime(int minutes)
        {
            int hours = minutes / 60;
            int mins = minutes % 60;
            return hours.ToString() + ":" + (mins < 10 ? "0" + mins.ToString() : mins.ToString());
        }

        //https://www.algoexpert.io/questions/generate-div-tags
        public static List<string> GenerateDivTags(int numberOfTags)
        {
            // O((2n)!/((n!((n + 1)!)))) time | O((2n)!/((n!((n + 1)!)))) space -
            // where n is the input number
            List<string> matchedDivTags = new List<string>();
            GenerateDivTagsFromPrefix(numberOfTags, numberOfTags, "", matchedDivTags);
            return matchedDivTags;
        }

        private static void GenerateDivTagsFromPrefix(int openingTagsNeeded, int closingTagsNeeded, string prefix, List<string> result)
        {
            if (openingTagsNeeded > 0)
            {
                string newPrefix = prefix + "<div>";
                GenerateDivTagsFromPrefix(openingTagsNeeded - 1, closingTagsNeeded, newPrefix, result);
            }

            if (openingTagsNeeded < closingTagsNeeded)
            {
                string newPrefix = prefix + "</div>";
                GenerateDivTagsFromPrefix(openingTagsNeeded, closingTagsNeeded - 1, newPrefix, result);
            }
            if (closingTagsNeeded == 0)
                result.Add(prefix);
        }
        //https://www.algoexpert.io/questions/best-seat
        public static int BestSeat(int[] seats)
        {
            //T:O(n) | S:O(1)
            int bestSeat = -1, maxSpace = 0;

            int left = 0; ;
            while (left < seats.Length - 1)
            {
                int right = left + 1;
                while (right < seats.Length && seats[right] == 0)
                    right++;

                int availableSpace = right - left + 1;
                if (availableSpace > maxSpace)
                {
                    bestSeat = (left + right) / 2;
                    maxSpace = availableSpace;
                }
                left = right;

            }
            return bestSeat;
        }
        //https://www.algoexpert.io/questions/number-of-ways-to-make-change 
        //Coins
        public static int NumberOfWaysToMakeChange(int n, int[] denoms)
        {

            //T: O(nd) | S: O(n) where d denotes denominations and n denotes amounts
            int[] ways = new int[n + 1];
            ways[0] = 1;

            foreach (int denom in denoms)
            {
                for (int amount = 0; amount < n + 1; amount++)
                {
                    if (denom <= amount)
                    {
                        ways[amount] += ways[amount - denom];
                    }

                }
            }
            return ways[n];
        }
        //https://www.algoexpert.io/questions/min-number-of-coins-for-change
        public static int MinimumNumberOfCoinsForChange(int n, int[] denoms)
        {
            //T:O(n*d) | S:O(n)
            int[] numOfCoins = new int[n + 1];
            Array.Fill(numOfCoins, Int32.MaxValue);
            numOfCoins[0] = 0;
            int toCompare = 0;
            foreach (int denom in denoms)
            {
                for (int amount = 0; amount <= numOfCoins.Length; amount++)
                {
                    if (denom <= amount)
                    {
                        if (numOfCoins[amount - denom] == Int32.MaxValue)
                        {
                            toCompare = numOfCoins[amount - denom];
                        }
                        else
                        {
                            toCompare = numOfCoins[amount - denom] + 1;
                        }
                        numOfCoins[amount] = Math.Min(numOfCoins[amount], toCompare);
                    }
                }
            }
            return numOfCoins[n] != int.MaxValue ? numOfCoins[n] : -1;

        }

        //https://www.algoexpert.io/questions/pattern-matcher
        public static string[] PatternMatacher(string pattern, string str)
        {
            //T:O(n^2+m) | S:O(n+m)
            if (pattern.Length > str.Length)
                return new string[] { };

            char[] newPattern = GetNewPattern(pattern);
            bool didSwitch = newPattern[0] != pattern[0];
            Dictionary<char, int> counts = new Dictionary<char, int>();
            counts['x'] = 0;
            counts['y'] = 0;
            int firstYPos = GetCountsAndFirstYPos(newPattern, counts);
            if (counts['y'] != 0)
            {
                for (int lenOfX = 1; lenOfX < str.Length; lenOfX++)
                {
                    double lenOfY = (double)(str.Length - lenOfX * counts['x']) / counts['y'];

                    if (lenOfY <= 0 || lenOfY % 1 != 0)
                        continue;

                    int yIdx = firstYPos * lenOfX;
                    string x = str.Substring(0, lenOfX);
                    string y = str.Substring(yIdx, (int)lenOfY);
                    string potentialMatch = BuildPotentialMatch(newPattern, x, y);
                    if (str.Equals(potentialMatch))
                    {
                        return didSwitch ? new string[] { y, x } : new string[] { x, y };
                    }
                }
            }
            else
            {
                double lenOfX = str.Length / counts['x'];
                if (lenOfX % 1 == 0)
                {
                    string x = str.Substring(0, (int)lenOfX);
                    string potentialMatch = BuildPotentialMatch(newPattern, x, "");
                    if (str.Equals(potentialMatch))
                    {
                        return didSwitch ? new string[] { "", x } : new string[] { x, "" };
                    }
                }

            }
            return new string[] { };
        }

        private static string BuildPotentialMatch(char[] newPattern, string x, string y)
        {
            StringBuilder potentialMatch = new StringBuilder();
            foreach (char c in newPattern)
            {
                if (c == 'x')
                {
                    potentialMatch.Append(x);
                }
                else potentialMatch.Append(y);
            }
            return potentialMatch.ToString();
        }

        private static int GetCountsAndFirstYPos(char[] pattern, Dictionary<char, int> counts)
        {
            int firstYPos = -1;
            for (int i = 0; i < pattern.Length; i++)
            {
                counts[pattern[i]]++;
                if (pattern[i] == 'Y' && firstYPos == -1)
                {
                    firstYPos = i;
                }
            }
            return firstYPos;
        }

        private static char[] GetNewPattern(string pattern)
        {
            char[] patternLetters = pattern.ToCharArray();
            if (pattern[0] == 'x') return patternLetters;
            for (int i = 0; i < patternLetters.Length; i++)
            {
                if (patternLetters[i] == 'x')
                    patternLetters[i] = 'y';
                else
                    patternLetters[i] = 'x';

            }
            return patternLetters;
        }

        public class Meeting
        {
            public int Start { get; set; }
            public int End { get; set; }

            public Meeting(int start, int end)
            {
                this.Start = start;
                this.End = end;
            }
        }
        public class StringMeeting
        {
            public string Start { get; set; }
            public string End { get; set; }

            public StringMeeting(string start, string end)
            {
                this.Start = start;
                this.End = end;
            }
        }
        //https://www.algoexpert.io/questions/search-for-range
        // O(log(n)) time | O(log(n)) space
        public static int[] SearchForRangeNonOptimal(int[] array, int target)
        {
            int[] finalRange = { -1, -1 };
            alteredBinarySearchNonOptimal(array, target, 0, array.Length - 1, finalRange, true);
            alteredBinarySearchNonOptimal(array, target, 0, array.Length - 1, finalRange, false);
            return finalRange;
        }

        public static void alteredBinarySearchNonOptimal(
            int[] array, int target, int left, int right, int[] finalRange, bool goLeft
        )
        {
            if (left > right)
            {
                return;
            }
            int mid = (left + right) / 2;
            if (array[mid] < target)
            {
                alteredBinarySearchNonOptimal(array, target, mid + 1, right, finalRange, goLeft);
            }
            else if (array[mid] > target)
            {
                alteredBinarySearchNonOptimal(array, target, left, mid - 1, finalRange, goLeft);
            }
            else
            {
                if (goLeft)
                {
                    if (mid == 0 || array[mid - 1] != target)
                    {
                        finalRange[0] = mid;
                    }
                    else
                    {
                        alteredBinarySearchNonOptimal(array, target, left, mid - 1, finalRange, goLeft);
                    }
                }
                else
                {
                    if (mid == array.Length - 1 || array[mid + 1] != target)
                    {
                        finalRange[1] = mid;
                    }
                    else
                    {
                        alteredBinarySearchNonOptimal(
                            array, target, mid + 1, right, finalRange, goLeft
                        );
                    }
                }
            }
        }

        // O(log(n)) time | O(1) space
        public static int[] SearchForRangeOptimal(int[] array, int target)
        {
            int[] finalRange = { -1, -1 };
            alteredBinarySearchOptimal(array, target, 0, array.Length - 1, finalRange, true);
            alteredBinarySearchOptimal(array, target, 0, array.Length - 1, finalRange, false);
            return finalRange;
        }

        public static void alteredBinarySearchOptimal(
            int[] array, int target, int left, int right, int[] finalRange, bool goLeft
        )
        {
            while (left <= right)
            {
                int mid = (left + right) / 2;
                if (array[mid] < target)
                {
                    left = mid + 1;
                }
                else if (array[mid] > target)
                {
                    right = mid - 1;
                }
                else
                {
                    if (goLeft)
                    {
                        if (mid == 0 || array[mid - 1] != target)
                        {
                            finalRange[0] = mid;
                            return;
                        }
                        else
                        {
                            right = mid - 1;
                        }
                    }
                    else
                    {
                        if (mid == array.Length - 1 || array[mid + 1] != target)
                        {
                            finalRange[1] = mid;
                            return;
                        }
                        else
                        {
                            left = mid + 1;
                        }
                    }
                }
            }
        }
        //https://www.algoexpert.io/questions/dice-throws
        public static int DiceThrows(int numSides, int numDices, int target)
        {

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
            int[,] storedResults = new int[2, target + 1];
            storedResults[0, 0] = 1;

            int prevousNumDiceIndex = 0;
            int newNumDiceIndex = 1;
            for (int currentNumDice = 0; currentNumDice < numDices; currentNumDice++)
            {
                for (int currentTarget = 0; currentTarget <= target; currentTarget++)
                {
                    int numWaysToReachTarget = 0;
                    for (int currentNumSides = 1;
                            currentNumSides <= Math.Min(currentTarget, numSides);
                            currentNumSides++)
                    {

                        numWaysToReachTarget += storedResults[prevousNumDiceIndex, currentTarget - currentNumSides];

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
            int[,] storedResults = new int[numSides + 1, target + 1];
            storedResults[0, 0] = 1;

            for (int currentNumDice = 1; currentNumDice <= numDices; currentNumDice++)
            {
                for (int currentTarget = 0; currentTarget <= target; currentTarget++)
                {
                    int numWaysToReachTarget = 0;
                    for (int currentNumSides = 1; currentNumSides <= Math.Min(currentTarget, numSides);
                            currentNumSides++)
                    {
                        numWaysToReachTarget += storedResults[currentNumDice - 1, currentTarget - currentNumSides];
                    }
                    storedResults[currentNumDice, currentTarget] = numWaysToReachTarget;
                }
            }
            return storedResults[numDices, target];
        }

        private static int DiceThrowsRecursive(int numSides, int numDices, int target)
        {
            int[,] storedResults = new int[numSides + 1, target + 1];

            for (int row = 0; row < storedResults.GetLength(0); row++)
            {
                for (int col = 0; col < storedResults.GetLength(1); col++)
                {
                    storedResults[row, col] = -1;
                }
            }
            return DiceThrowsRecursiveHelper(numDices, numSides, target, storedResults);
        }

        private static int DiceThrowsRecursiveHelper(int numDices, int numSides, int target, int[,] storedResults)
        {
            if (numDices == 0)
            { //DT(0,0)
                return target == 0 ? 1 : 0;
            }
            if (storedResults[numDices, target] != -1)
            {
                return storedResults[numDices, target];
            }
            int numWaysToReachTarget = 0;
            for (int currentTarget = Math.Max(0, target - numSides);
                    currentTarget < target; currentTarget++)
            {

                numWaysToReachTarget += DiceThrowsRecursiveHelper(numDices - 1, numSides, currentTarget, storedResults);
            }
            storedResults[numDices, target] = numWaysToReachTarget;
            return numWaysToReachTarget;
        }
        //https://www.algoexpert.io/questions/disk-stacking
        public static List<int[]> DiskStacking(List<int[]> disks)
        {
            //1.T:O(n^2) | S:O(n)
            disks.Sort((disk1, disk2) => disk1[2].CompareTo(disk2[2]));
            int[] heights = new int[disks.Count];
            for (int i = 0; i < disks.Count; i++)
            {
                heights[i] = disks[i][2];
            }
            int[] sequences = new int[disks.Count];
            for (int i = 0; i < disks.Count; i++)
            {
                sequences[i] = Int32.MinValue;
            }

            int maxHeightIdx = 0;
            for (int i = 1; i < disks.Count; i++)
            {
                int[] currentDisk = disks[i];
                for (int j = 0; j < i; j++)
                {
                    int[] otherDisk = disks[j];
                    if (areValidDimentions(otherDisk, currentDisk))
                    {
                        if (heights[i] <= currentDisk[2] + heights[j])
                        {
                            heights[i] = currentDisk[2] + heights[j];
                            sequences[i] = j;
                        }
                    }
                }
                if (heights[i] >= heights[maxHeightIdx])
                    maxHeightIdx = heights[i];
            }
            return BuildSequence(disks, sequences, maxHeightIdx);

        }

        private static List<int[]> BuildSequence(List<int[]> disks, int[] sequences, int currentIdx)
        {
            List<int[]> seq = new List<int[]>();
            while (currentIdx != Int32.MinValue)
            {
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
        public static int[][][] KrusaklMST(int[][][] edges)
        {

            //T:O(e*log(e)) | S: O(e+v)

            List<List<int>> sortedEdges = new List<List<int>>();
            for (int sourceIndex = 0; sourceIndex < edges.Length; sourceIndex++)
            {
                foreach (var edge in edges[sourceIndex])
                {
                    //skip reverse edges as this is an undirected graph
                    if (edge[0] > sourceIndex)
                    {
                        sortedEdges.Add(new List<int>() { sourceIndex, edge[0], edge[1] });
                    }
                }
            }
            sortedEdges.Sort((edge1, edge2) => edge1[2] - edge2[2]);
            int[] parents = new int[edges.Length];
            int[] ranks = new int[edges.Length];
            List<List<int[]>> mst = new List<List<int[]>>();

            for (int i = 0; i < edges.Length; i++)
            {
                parents[i] = i;
                ranks[i] = 0;
                mst.Insert(i, new List<int[]>());
            }

            foreach (var edge in sortedEdges)
            {
                int vertex1Root = Find(edge[0], parents);
                int vertex2Root = Find(edge[1], parents);

                if (vertex1Root != vertex2Root)
                {
                    mst[edge[0]].Add(new int[] { edge[1], edge[2] });
                    mst[edge[1]].Add(new int[] { edge[0], edge[2] });

                    Union(vertex1Root, vertex2Root, parents, ranks);
                }

            }

            int[][][] arrayMst = new int[edges.Length][][];
            for (int i = 0; i < mst.Count; i++)
            {
                arrayMst[i] = new int[mst[i].Count][];
                for (int j = 0; j < mst[i].Count; j++)
                {
                    arrayMst[i][j] = mst[i][j];
                }
            }
            return arrayMst;
        }

        private static void Union(int vertex1Root, int vertex2Root, int[] parents, int[] ranks)
        {
            if (ranks[vertex1Root] < ranks[vertex2Root])
            {
                parents[vertex1Root] = vertex2Root;
            }
            else if (ranks[vertex1Root] > ranks[vertex2Root])
            {
                parents[vertex2Root] = vertex1Root;
            }
            else
            {
                parents[vertex2Root] = vertex1Root;
                ranks[vertex1Root]++;
            }
        }

        private static int Find(int vertex, int[] parents)
        {
            if (vertex != parents[vertex])
            {
                parents[vertex] = Find(parents[vertex], parents); //Path Compression
            }
            return parents[vertex];
        }
        //https://www.algoexpert.io/questions/shorten-path
        //Ex: /foo/../test/../test/../foo//bar/./baz =>  /foo/bar/baz
        public static string ShortenPath(string path)
        {
            //T:O(n)| S:O(n) - n is lenght of path
            bool startsWithPath = path[0] == '/';
            string[] tokensArr = path.Split("/");
            List<string> tokenList = new List<string>(tokensArr);
            List<string> filteredTokens = tokenList.FindAll(token => IsImportantToken(token));
            Stack<string> stack = new Stack<string>();
            if (startsWithPath) stack.Push("");
            foreach (string token in filteredTokens)
            {
                if (token.Equals(".."))
                {
                    if (stack.Count == 0 || stack.Peek().Equals(".."))
                    {
                        stack.Push(token);
                    }
                    else if (!stack.Peek().Equals(""))
                    {
                        stack.Pop();
                    }
                }
                else
                {
                    stack.Push(token);
                }

            }
            if (stack.Count == 1 && stack.Peek().Equals("")) return "/";
            var arr = stack.ToArray();
            Array.Reverse(arr);
            return string.Join("/", arr);

        }

        private static bool IsImportantToken(string token)
        {
            return token.Length > 0 && !token.Equals(".");
        }
        //https://www.algoexpert.io/questions/sweet-and-savory
        public static int[] SweetAndSavory(int[] dishes, int target)
        {

            //T:O(nlog(n)) | S:O(n)
            List<int> sweetDishes = new List<int>();
            List<int> savoryDishes = new List<int>();

            foreach (var dish in dishes)
            {
                if (dish < 0)
                    sweetDishes.Add(dish);
                else
                    savoryDishes.Add(dish);

            }
            sweetDishes.Sort((a, b) => Math.Abs(a) - Math.Abs(b));
            savoryDishes.Sort();

            int[] bestPair = new int[2];
            int bestDiff = Int32.MinValue;
            int sweetIndx = 0, savoryIndex = 0;

            while (sweetIndx < sweetDishes.Count && savoryIndex < savoryDishes.Count)
            {
                int currentSum = sweetDishes[sweetIndx] + savoryDishes[savoryIndex];

                if (currentSum <= target)
                {
                    int currentDiff = target - currentSum;
                    if (currentDiff < bestDiff)
                    {
                        bestDiff = currentDiff;
                        bestPair[0] = sweetDishes[sweetIndx];
                        bestPair[1] = savoryDishes[savoryIndex];
                    }
                    savoryIndex++;
                }
                else sweetIndx++;
            }
            return bestPair;

        }
        //https://www.algoexpert.io/questions/stable-internships
        //Gale–Shapley algorithm or Stable match or Stable marriage algo
        public int[][] StableInternships(int[][] interns, int[][] teams)
        {
            //T:O(n^2) | S:O(n^2)
            Dictionary<int, int> choosenInterns = new Dictionary<int, int>();
            Stack<int> freeInterns = new Stack<int>();
            for (int i = 0; i < interns.Length; i++)
                freeInterns.Push(i);

            int[] currentInternChoices = new int[interns.Length];

            List<Dictionary<int, int>> teamDict = new List<Dictionary<int, int>>();
            foreach (var team in teams)
            {
                Dictionary<int, int> rank = new Dictionary<int, int>();
                for (int i = 0; i < team.Length; i++)
                {
                    rank[team[i]] = i;
                }
                teamDict.Add(rank);
            }
            while (freeInterns.Count > 0)
            {

                int internNum = freeInterns.Pop();

                int[] intern = interns[internNum];
                int teamPref = intern[currentInternChoices[internNum]];
                currentInternChoices[internNum]++;

                if (!choosenInterns.ContainsKey(teamPref))
                {
                    choosenInterns[teamPref] = internNum;
                    continue;
                }

                int prevIntern = choosenInterns[teamPref];
                int prevInternRank = teamDict[teamPref][prevIntern];
                int currentInternRank = teamDict[teamPref][internNum];

                if (currentInternRank < prevInternRank)
                {
                    freeInterns.Push(prevIntern);
                    choosenInterns[teamPref] = internNum;
                }
                else
                {
                    freeInterns.Push(internNum);
                }
            }
            int[][] matches = new int[interns.Length][];
            int index = 0;
            foreach (var choosenIntern in choosenInterns)
            {
                matches[index] = new int[] { choosenIntern.Value, choosenIntern.Key };
                index++;
            }
            return matches;
        }

        //https://www.algoexpert.io/questions/group-anagrams
        public static List<List<string>> GroupAnagrams(List<string> words)
        {

            if (words.Count == 0) return new List<List<string>>();
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

            foreach (string word in words)
            {
                char[] chArr = word.ToCharArray();
                Array.Sort(chArr);
                string sorteWord = new string(chArr);
                if (anagrams.ContainsKey(sorteWord))
                {
                    anagrams[sorteWord].Add(word);
                }
                else
                {
                    anagrams[sorteWord] = new List<string>() { word };
                }
            }

            return anagrams.Values.ToList();
        }

        private static List<List<string>> GroupAnagramsNaive(List<string> words)
        {
            List<string> sortedWords = new List<string>();
            foreach (string word in words)
            {
                char[] chArr = word.ToCharArray();
                Array.Sort(chArr);
                string sortedWord = new string(chArr);
                sortedWords.Add(sortedWord);
            }

            List<int> indices = Enumerable.Range(0, words.Count).ToList();
            indices.Sort((a, b) => sortedWords[a].CompareTo(sortedWords[b]));

            List<List<string>> result = new List<List<string>>();
            List<string> currentAnagramGrp = new List<string>();
            string currentAnagram = sortedWords[indices[0]];
            foreach (int index in indices)
            {
                string word = words[index];
                string sortedWord = sortedWords[index];

                if (sortedWord.Equals(currentAnagram))
                {
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
        public static List<string> ValidIPAddresses(string str)
        {

            //T:O(1) as in 2^32 IP addresses for full 12 digit | S:O(1)

            List<string> ipAddresessFound = new List<string>();

            for (int i = 1; i < Math.Min((int)str.Length, 4); i++)
            {
                string[] currentIPAddressParts = new string[] { "", "", "", "" };

                currentIPAddressParts[0] = str.Substring(0, i - 0);
                if (!IsValidPart(currentIPAddressParts[0]))
                {
                    continue;
                }
                for (int j = i + 1; j < i + Math.Min((int)str.Length - i, 4); j++)
                { //or j< Math.Min((int)str.Length, i+4)
                    currentIPAddressParts[1] = str.Substring(i, j - i);
                    if (!IsValidPart(currentIPAddressParts[1]))
                        continue;

                    for (int k = j + 1; k < j + Math.Min((int)str.Length - j, 4); k++)
                    {
                        currentIPAddressParts[2] = str.Substring(j, k - j);
                        currentIPAddressParts[3] = str.Substring(k);

                        if (IsValidPart(currentIPAddressParts[2]) && IsValidPart(currentIPAddressParts[3]))
                        {
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
            for (int i = 0; i < currentIPAddressParts.Length; i++)
            {
                sb.Append(currentIPAddressParts[i]);
                if (i < currentIPAddressParts.Length)
                    sb.Append(".");
            }
            return sb.ToString();
        }

        private static bool IsValidPart(string str)
        {
            int stringAsInt = Int32.Parse(str);
            if (stringAsInt > 255) return false;
            return str.Length == stringAsInt.ToString().Length; //Check for leading 0's
        }

        //https://www.algoexpert.io/questions/staircase-traversal
        public static int StaircaseTraversal(int height, int maxSteps)
        {


            //1.Naive - Recursion
            //T:O(k^n) - exponetial | S:O(n) where n is height of staircase and k is number of allowed steps
            int numberOfWaysToTop = NumberOfWaysToTopNaiveRec(height, maxSteps);

            //2.Optimal - Recursion with Memorization (DP)
            //T:O(k*n) | S:O(n) where n is height of staircase and k is number of allowed steps
            Dictionary<int, int> memoize = new Dictionary<int, int>();
            memoize[0] = 1;
            memoize[1] = 1;
            numberOfWaysToTop = NumberOfWaysToTopOptimalRec(height, maxSteps, memoize);

            //3.Optimal - Iterative with Memorization (DP)
            //T:O(k*n) | S:O(n) where n is height of staircase and k is number of allowed steps
            numberOfWaysToTop = NumberOfWaysToTopOptimalIterative(height, maxSteps);

            //4.Optimal - Iterative with Memorization (DP)  
            //T:O(n) | S:O(n) where n is height of staircase and k is number of allowed steps
            numberOfWaysToTop = NumberOfWaysToTopOptimalIterative2(height, maxSteps);

            return numberOfWaysToTop;
        }

        private static int NumberOfWaysToTopOptimalIterative2(int height, int maxSteps)
        {
            int curNumOfWays = 0;
            List<int> waysToTop = new List<int>();
            waysToTop.Add(1);
            for (int curHeight = 1; curHeight < height + 1; curHeight++)
            {

                int startOfWindow = curHeight - maxSteps - 1;
                int endOfWindow = curHeight - 1;

                if (startOfWindow >= 0)
                {
                    curNumOfWays -= waysToTop[startOfWindow];
                }
                curNumOfWays += waysToTop[endOfWindow];
                waysToTop.Add(curNumOfWays);
            }
            return waysToTop[height];

        }

        private static int NumberOfWaysToTopOptimalIterative(int height, int maxSteps)
        {
            int[] waysToTop = new int[height + 1];
            waysToTop[0] = 1;
            waysToTop[1] = 1;
            for (int curHeight = 2; curHeight < height + 1; curHeight++)
            {
                int step = 1;
                while (step <= maxSteps && step <= curHeight)
                {
                    waysToTop[curHeight] = waysToTop[curHeight] + waysToTop[curHeight - step];
                    step++;
                }

            }
            return waysToTop[height];

        }

        private static int NumberOfWaysToTopOptimalRec(int height, int maxSteps, Dictionary<int, int> memoize)
        {
            if (memoize.ContainsKey(height))
                return memoize[height];

            int numberOfWays = 0;

            for (int step = 1; step < Math.Min(maxSteps, height) + 1; step++)
            {
                numberOfWays += NumberOfWaysToTopOptimalRec(height - step, maxSteps, memoize);
            }
            memoize[height] = numberOfWays;

            return numberOfWays;

        }

        private static int NumberOfWaysToTopNaiveRec(int height, int maxSteps)
        {
            if (height <= 1)
                return 1;
            int numberOfWays = 0;
            for (int step = 1; step < Math.Min(maxSteps, height) + 1; step++)
            {
                numberOfWays += NumberOfWaysToTopNaiveRec(height - step, maxSteps);
            }
            return numberOfWays;
        }
        //https://www.algoexpert.io/questions/reversePolishNotation
        public static int ReversePolishNotation(string[] tokens)
        {
            Stack<int> operands = new Stack<int>();
            foreach (string token in tokens)
            {
                if (token.Equals("+"))
                {
                    operands.Push(operands.Pop() + operands.Pop());
                }
                else if (token.Equals("-"))
                {
                    int firstNum = operands.Pop();
                    operands.Push(operands.Pop() - firstNum);
                }
                else if (token.Equals("*"))
                {
                    operands.Push(operands.Pop() * operands.Pop());
                }
                else if (token.Equals("/"))
                {
                    int firstNum = operands.Pop();
                    operands.Push(operands.Pop() / firstNum);
                }
                else
                {
                    operands.Push(Int32.Parse(token));
                }
            }
            return operands.Pop();
        }

        //https://www.algoexpert.io/questions/evaluate-expression-tree
        public int EvaluateExpressionTree(BinaryTree tree)
        {
            //T:O(n) time | S: O(h) space - where n is the number of nodes in the Binary Tree, and h is the height of the Binary Tree
            return EvalExpTree(tree);
        }
        private int EvalExpTree(BinaryTree node)
        {

            if (node.Value >= 0) return node.Value;

            int left = EvalExpTree(node.Left);
            int right = EvalExpTree(node.Right);
            int res = EvalExp(left, right, node.Value);

            return res;

        }
        private int EvalExp(int op1, int op2, int opera)
        {
            if (opera == -1)
                return op1 + op2;
            else if (opera == -2)
                return op1 - op2;
            else if (opera == -3)
                return op1 / op2;
            else if (opera == -4)
                return op1 * op2;
            else
                return opera;
        }

        //https://www.algoexpert.io/questions/water-area
        //rain water trapped
        public static int WaterArea(int[] heights)
        {

            if (heights.Length == 0) return 0;
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
            int leftIdx = 0, rightIdx = heights.Length - 1;
            int leftMax = heights[leftIdx], rightMax = heights[rightIdx];
            var trappedWater = 0;

            while (leftIdx < rightIdx)
            {
                if (heights[leftIdx] < heights[rightIdx])
                {
                    leftIdx++;
                    leftMax = Math.Max(leftMax, heights[leftIdx]);
                    trappedWater += leftMax - heights[leftIdx];
                }
                else
                {
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
            int leftMax = 0;
            for (int i = 0; i < heights.Length; i++)
            {
                int height = heights[i];
                maxes[i] = leftMax;
                leftMax = Math.Max(leftMax, height);
            }

            int rightMax = 0;
            for (int i = heights.Length - 1; i >= 0; i--)
            {
                int height = heights[i];
                int minHeight = Math.Min(maxes[i], rightMax);
                if (height < minHeight)
                {
                    maxes[i] = minHeight - height;
                }
                else maxes[i] = 0;

                rightMax = Math.Max(rightMax, height);

            }
            int total = 0;
            for (int idx = 0; idx < maxes.Length; idx++)
            {
                total += maxes[idx];
            }
            return total;
        }
        //https://www.algoexpert.io/questions/optimal-freelancing
        public static int OptimalFreelancing(Dictionary<string, int>[] jobs)
        {
            //1.Naive - with pair of loops to compare max profit against each combination
            //T:O(n^2) : SO(1)

            //2.T:O(nlogn) | S:O(1)
            const int LENGTH_OF_PERIOD = 7;

            int maxProfit = 0;


            Array.Sort(
                    jobs,
                    Comparer<Dictionary<string, int>>.Create(
                        (jobOne, jobTwo) => jobTwo["payment"].CompareTo(jobOne["payment"])
                        )
                    );

            bool[] timeLine = new bool[LENGTH_OF_PERIOD];
            foreach (var job in jobs)
            {
                int maxTime = Math.Min(job["deadline"], LENGTH_OF_PERIOD);
                for (int time = maxTime - 1; time >= 0; time--)
                {
                    if (!timeLine[time])
                    {
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

        public static int[][][] PrimsMST(int[][][] edges)
        {
            //T:O(e*log(v)) | S:O(v+e) - e is number of edges and v is number of vertices.
            List<Item> initialEdgeItems = new List<Item>();
            foreach (var edge in edges[0])
            {
                Item edgeItem = new Item(0, edge[0], edge[1]);
                initialEdgeItems.Add(edgeItem);
            }
            Heap<Item> minHeap = new Heap<Item>(initialEdgeItems, CompareByDistance);

            List<List<int[]>> mst = new List<List<int[]>>();
            for (int i = 0; i < edges.Length; i++)
            {
                mst.Add(new List<int[]>());
            }
            while (!minHeap.IsEmpty())
            {
                Item heapItem = minHeap.Remove();
                int vertex = heapItem.Vertex;
                int discoverdVertex = heapItem.DiscoverdVertex;
                int distance = heapItem.Distance;

                if (mst[discoverdVertex].Count == 0)
                {
                    mst[vertex].Add(new int[] { discoverdVertex, distance });
                    mst[discoverdVertex].Add(new int[] { vertex, distance });

                    foreach (var edge in edges[discoverdVertex])
                    {
                        int neighbor = edge[0];
                        int neighborDistance = edge[1];

                        if (mst[neighbor].Count == 0)
                        {
                            minHeap.Insert(new Item(discoverdVertex, neighbor, neighborDistance));
                        }
                    }
                }
            }
            int[][][] arrayMST = new int[edges.Length][][];
            for (int i = 0; i < mst.Count; i++)
            {
                arrayMST[i] = new int[mst[i].Count][];
                for (int j = 0; j < mst[i].Count; j++)
                {
                    arrayMST[i][j] = mst[i][j];
                }
            }
            return arrayMST;
        }
        private static bool CompareByDistance(Item item1, Item item2)
        {

            return item1.Distance < item2.Distance;
        }
        class Item
        {
            public int Vertex;
            public int DiscoverdVertex;
            public int Distance;

            public Item(int vertex, int discoverdVertex, int distance)
            {
                this.Vertex = vertex;
                this.DiscoverdVertex = discoverdVertex;
                this.Distance = distance;
            }
        }
        //https://www.algoexpert.io/questions/topological-sort
        public static List<int> TopologicalSort(List<int> jobs, List<int[]> deps)
        {

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
            JobGraph jobGraph = new JobGraph(jobs);
            foreach (int[] dep in deps)
            {
                jobGraph.AddDep(dep[0], dep[1]);
            }
            return jobGraph;
        }
        private static List<int> GetOrderedJobs1(JobGraph jobGraph)
        {
            List<int> orderedJobs = new List<int>();
            Stack<JobNode> nodesWithNoPrereqs = new Stack<JobNode>();
            foreach (JobNode node in jobGraph.nodes)
            {
                if (node.NumOfPrereqs == 0) nodesWithNoPrereqs.Push(node);
            }
            while (nodesWithNoPrereqs.Count > 0)
            {
                JobNode node = nodesWithNoPrereqs.Pop();
                orderedJobs.Add(node.Job);
                RemoveDeps(node, nodesWithNoPrereqs);
            }
            bool graphHasEdges = false; //Cycle check
            foreach (JobNode jobNode in jobGraph.nodes)
            {
                if (jobNode.NumOfPrereqs > 0) graphHasEdges = true;
            }
            return graphHasEdges ? new List<int>() : orderedJobs;
        }

        private static void RemoveDeps(JobNode node, Stack<JobNode> nodesWithNoPrereqs)
        {
            while (node.Deps.Count > 0)
            {
                JobNode dep = node.Deps[node.Deps.Count - 1];
                node.Deps.RemoveAt(node.Deps.Count - 1);
                dep.NumOfPrereqs--;
                if (dep.NumOfPrereqs == 0) nodesWithNoPrereqs.Push(dep);
            }
        }

        private static List<int> GetOrderedJobs(JobGraph jobGraph)
        {
            List<int> orderedJobs = new List<int>();
            Stack<JobNode> nodeStack = jobGraph.nodes;
            while (nodeStack.Count > 0)
            {
                JobNode node = nodeStack.Pop();
                bool containsCycle = DepthFirstTraverse(node, orderedJobs);
                if (containsCycle) return new List<int>();

            }
            return orderedJobs;
        }

        private static bool DepthFirstTraverse(JobNode node, List<int> orderedJobs)
        {
            if (node.Visited) return false;
            if (node.Visiting) return true;
            node.Visiting = true;
            foreach (JobNode prereqNode in node.Prereqs)
            {
                bool containsCycle = DepthFirstTraverse(prereqNode, orderedJobs);
                if (containsCycle) return true;
            }
            node.Visited = true;
            node.Visiting = false;
            orderedJobs.Add(node.Job);
            return false;
        }

        private static JobGraph CreateJobGraph(List<int> jobs, List<int[]> deps)
        {
            JobGraph jobGraph = new JobGraph(jobs);
            foreach (int[] dep in deps)
            {
                jobGraph.AddPrereq(dep[1], dep[0]);
            }
            return jobGraph;
        }
        public class JobGraph
        {
            public Stack<JobNode> nodes;
            public Dictionary<int, JobNode> graph;

            public JobGraph(List<int> jobs)
            {
                nodes = new Stack<JobNode>();
                graph = new Dictionary<int, JobNode>();
                foreach (int job in jobs)
                {
                    AddNode(job);
                }
            }
            private void AddNode(int job)
            {
                graph.Add(job, new JobNode(job));
                nodes.Push(graph[job]);

            }
            public void AddPrereq(int job, int prereq)
            {
                JobNode jobNode = GetNode(job);
                JobNode prereqNode = GetNode(prereq);
                jobNode.Prereqs.Add(prereqNode);

            }

            public void AddDep(int job, int dep)
            {  //For apporach#2
                JobNode jobNode = GetNode(job);
                JobNode depNode = GetNode(dep);
                jobNode.Deps.Add(depNode);
                depNode.NumOfPrereqs++;

            }

            private JobNode GetNode(int job)
            {
                if (!graph.ContainsKey(job)) AddNode(job);
                return graph[job];
            }
        }
        public class JobNode
        {
            public int Job;
            public List<JobNode> Prereqs;
            public List<JobNode> Deps;  //for Approach 2
            public bool Visited;
            public bool Visiting;
            public int NumOfPrereqs;
            public JobNode(int job)
            {
                this.Job = job;
                this.Prereqs = new List<JobNode>();
                Visited = false;
                Visiting = false;
                NumOfPrereqs = 0;
            }

        }
        //https://www.algoexpert.io/questions/juice-bottling
        public static List<int> JuiceBottling(int[] prices)
        {

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

            for (int size = 0; size < numSizes; size++)
            {
                for (int dividingPoint = 0; dividingPoint <= size; dividingPoint++)
                {
                    int possibleProfit = maxProfit[size - dividingPoint] + prices[dividingPoint];

                    if (possibleProfit > maxProfit[size])
                    {
                        maxProfit[size] = possibleProfit;
                        dividingPoints[size] = dividingPoint;
                    }
                }
            }
            List<int> solution = new List<int>();
            int curDividingPoint = numSizes - 1;
            while (curDividingPoint > 0)
            {
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
            for (int size = 0; size < numSizes; size++)
            {
                solutions.Add(new List<int>());
            }
            for (int size = 0; size < numSizes; size++)
            { //O(n)
                for (int dividingPoint = 0; dividingPoint <= size; dividingPoint++)
                { //O(n)
                    int possibleProfit = maxProfilt[size - dividingPoint] + prices[dividingPoint];

                    if (possibleProfit > maxProfilt[size])
                    {
                        maxProfilt[size] = possibleProfit;
                        List<int> newSolution = new List<int>();
                        newSolution.Add(dividingPoint);
                        newSolution.AddRange(solutions[size - dividingPoint]); ////O(n)
                        solutions[size] = newSolution;
                    }
                }
            }
            return solutions[numSizes - 1];
        }

        //https://www.algoexpert.io/questions/detect-arbitrage
        public static bool DetectArbitrage(List<List<Double>> exchangeRates)
        {

        TODO:
            // O(n^3) time | O(n^2) space - where n is the number of currencies

            // To use exchange rates as edge weights, we must be able to add them.
            // Since log(a*b) = log(a) + log(b), we can convert all rates to
            // -log10(rate) to use them as edge weights.
            List<List<Double>> logExchangeRates = ConvertToLogMatrix(exchangeRates);

            // A negative weight cycle indicates an arbitrage.
            return FoundNegativeWeightCycle(logExchangeRates, 0);
        }

        // Runs the Bellman–Ford Algorithm to detect any negative-weight cycles.
        private static bool FoundNegativeWeightCycle(List<List<double>> graph, int start)
        {
            double[] distancesFromStart = new double[graph.Count];
            Array.Fill(distancesFromStart, Double.MaxValue);
            distancesFromStart[start] = 0;

            for (int unused = 0; unused < graph.Count; unused++)
            {
                // If no update occurs, that means there's no negative cycle.
                if (!RelaxEdgesAndUpdateDistances(graph, distancesFromStart))
                {
                    return false;
                }
            }

            return RelaxEdgesAndUpdateDistances(graph, distancesFromStart);
        }

        private static List<List<double>> ConvertToLogMatrix(List<List<double>> matrix)
        {
            List<List<Double>> newMatrix = new List<List<Double>>();

            for (int row = 0; row < matrix.Count; row++)
            {
                List<Double> rates = matrix[row];
                newMatrix.Add(new List<Double>());
                foreach (var rate in rates)
                {
                    newMatrix[row].Add(-Math.Log10(rate));
                }
            }
            return newMatrix;
        }



        private static bool RelaxEdgesAndUpdateDistances(List<List<double>> graph, double[] distances)
        {
            bool updated = false;

            for (int sourceIdx = 0; sourceIdx < graph.Count; sourceIdx++)
            {
                List<Double> edges = graph[sourceIdx];
                for (int destinationIdx = 0; destinationIdx < edges.Count;
                    destinationIdx++)
                {
                    double edgeWeight = edges[destinationIdx];
                    double newDistanceToDestination = distances[sourceIdx] + edgeWeight;
                    if (newDistanceToDestination < distances[destinationIdx])
                    {
                        updated = true;
                        distances[destinationIdx] = newDistanceToDestination;
                    }
                }
            }

            return updated;
        }
        //https://www.algoexpert.io/questions/task-assignment
        public static List<List<int>> TaskAssignment(int k, List<int> tasks)
        {
            //1.Naive - Generate all possible pairs and check for optimal duration
            //T:O(n^2)  | S:O(1)

            //2.Sorting to join largest duration with smallest duration inorder to achieve optimal duration 
            //T:O(nlogn) | S:O(n)
            List<List<int>> pairedTasks = new List<List<int>>();
            Dictionary<int, List<int>> taskDurationsToIndices = GetTaskDurationsToIndices(tasks);

            tasks.Sort();

            for (int idx = 0; idx < k; idx++)
            {

                int task1Duration = tasks[0]; //Sorted one

                List<int> indicesWithTask1Duration = taskDurationsToIndices[task1Duration];
                int task1OriginalIndex = indicesWithTask1Duration[indicesWithTask1Duration.Count - 1];
                indicesWithTask1Duration.RemoveAt(indicesWithTask1Duration.Count - 1);

                int task2Index = tasks.Count - 1 - idx; //Sorted index
                int task2Duration = tasks[task2Index];
                List<int> indicesWithTask2Duration = taskDurationsToIndices[task2Duration];
                int task2OriginalIndex = indicesWithTask2Duration[indicesWithTask2Duration.Count - 1];
                indicesWithTask2Duration.RemoveAt(indicesWithTask2Duration.Count - 1);

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

            for (int idx = 0; idx < tasks.Count; idx++)
            {
                int taskDuration = tasks[idx];
                if (taskDurationsToIndices.ContainsKey(taskDuration))
                {
                    taskDurationsToIndices[taskDuration].Add(idx);
                }
                else
                {
                    List<int> temp = new List<int>();
                    temp.Add(idx);
                    taskDurationsToIndices[taskDuration] = temp;
                }
            }

            return taskDurationsToIndices;

        }
        //https://www.algoexpert.io/questions/sunset-views
        public static List<int> SunsetViews(int[] buildings, string direction)
        {

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


            int startIdx = buildings.Length - 1;
            int step = -1;

            if (direction.Equals("EAST"))
            {
                startIdx = 0;
                step = 1;
            }

            int idx = startIdx;
            while (idx >= 0 && idx < buildings.Length)
            {
                int buildingHeight = buildings[idx];

                while (candidateBuildings.Count > 0 && candidateBuildings.Peek() <= buildingHeight)
                {
                    candidateBuildings.Pop();
                }
                candidateBuildings.Push(idx);

                idx += step;
            }
            if (direction.Equals("WEST")) candidateBuildings.Reverse();

            return candidateBuildings.ToList();

        }

        private static List<int> SunsetViewsOptimal1(int[] buildings, string direction)
        {
            List<int> buildingsWithSunsetViews = new List<int>();

            int startIdx = buildings.Length - 1;
            int step = -1;

            if (direction.Equals("EAST"))
            {
                startIdx = 0;
                step = 1;
            }

            int idx = startIdx;
            int runningMaxHeight = 0;
            while (idx >= 0 && idx < buildings.Length)
            {
                int buildingHeight = buildings[idx];

                if (buildingHeight > runningMaxHeight)
                {
                    buildingsWithSunsetViews.Add(idx);
                }
                runningMaxHeight = Math.Max(runningMaxHeight, buildingHeight);

                idx += step;

            }
            if (direction.Equals("WEST")) buildingsWithSunsetViews.Reverse();

            return buildingsWithSunsetViews;
        }
        //https://www.algoexpert.io/questions/ambiguous-measurements
        public static bool AmbiguousMeasurements(int[][] measuringCups, int low, int high)
        {
            //DP Recursion with memoization
            //T:O(low*high*n) | S:O(low*high)                        
            /**Notes
As we recursively call our canMeasureInRange function, we might realize that if the input low is ever less than or equal to 0, the outcome for that low is always the same.
In other words, checking if we can measure in the range [-10, 10] is equivalent to checking if we can measure in the range [-5, 10], which itself is equivalent to checking if we can measure in the range [0, 10].
The same logic applies to the high value.
Thus, we can optimize the solution described in the video explanation by capping the low and high values that we pass to our canMeasureInRange function to 0. This reduces the number of keys in our cache and maximizes cache hits, thereby optimizing our solution in practice (though not from a time-complexity point of view).
The two comments in the code snippet below highlight the changes that we've made to the code covered in the video explanation.

// Change `<` to `<=`.
if low <= 0 and high <= 0:
    return False

canMeasure = False
for cup in measuringCups:
    cupLow, cupHigh = cup
    if low <= cupLow and cupHigh <= high:
        canMeasure = True
        break

    // Cap the `newLow` and `newHigh` to 0.
    newLow = max(0, low - cupLow)
    newHigh = max(0, high - cupHigh)
    canMeasure = canMeasureInRange(measuringCups, newLow, newHigh, memoization)
    if canMeasure:
        break
*/
            Dictionary<string, bool> memoization = new Dictionary<string, bool>();
            return CanMeasureInRange(measuringCups, low, high, memoization);

        }

        private static bool CanMeasureInRange(int[][] measuringCups, int low, int high, Dictionary<string, bool> memoization)
        {
            string memoizeKey = CreateHashTableKey(low, high);

            if (memoization.ContainsKey(memoizeKey))
                return memoization[memoizeKey];

            if (low <= 0 && high <= 0)
            {
                return false;
            }
            bool canMeasure = false;
            foreach (var cup in measuringCups)
            {
                int cupLow = cup[0];
                int cupHigh = cup[1];
                if (low <= cupLow && cupHigh <= high)
                {
                    canMeasure = true;
                    break;
                }
                int newLow = Math.Max(0, low - cupLow);
                int newHigh = Math.Max(0, high - cupHigh);
                canMeasure = CanMeasureInRange(measuringCups, newLow, newHigh, memoization);
                if (canMeasure) break;
            }
            memoization[memoizeKey] = canMeasure;
            return canMeasure;
        }

        private static string CreateHashTableKey(int low, int high)
        {
            return low.ToString() + "-" + high.ToString();
        }
        //https://www.algoexpert.io/questions/largest-park
        //Largest Rectangle in Histogram
        public static int LargestPark(bool[][] land)
        {
            //T:O(w*h) | S:O(w) - w and h are width(row) and height(column) of input matrix
            int[] heights = new int[land[0].Length];
            int maxArea = 0;
            foreach (var row in land)
            {
                for (int col = 0; col < land[0].Length; col++)
                {
                    heights[col] = row[col] == false ? heights[col] + 1 : 0;
                }
                maxArea = Math.Max(maxArea, LargestRectangleInHistogram(heights));
            }
            return maxArea;

        }

        private static int LargestRectangleInHistogram(int[] heights)
        {
            Stack<int> stack = new Stack<int>();
            int maxArea = 0;
            for (int col = 0; col < heights.Length; col++)
            {
                while (stack.Count > 0 && heights[col] < heights[stack.Peek()])
                {
                    int height = heights[stack.Pop()];
                    int width = (stack.Count == 0) ? col : col - stack.Peek() - 1;

                    maxArea = Math.Max(maxArea, width * height);

                }
                stack.Push(col);
            }
            //For remain elements
            while (stack.Count > 0)
            {
                int height = heights[stack.Pop()];
                int width = (stack.Count == 0) ? heights.Length : heights.Length - stack.Peek() - 1;
                maxArea = Math.Max(maxArea, height * width);
            }
            return maxArea;
        }
        //https://www.algoexpert.io/questions/laptop-rentals
        public static int LaptopRentals(List<List<int>> times)
        {
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
            if (times.Count == 0) return 0;

            int usedLaptops = 0;
            List<int> startTimes = new List<int>();
            List<int> endTimes = new List<int>();

            foreach (var interval in times)
            {
                startTimes.Add(interval[0]);
                endTimes.Add(interval[1]);
            }

            startTimes.Sort();
            endTimes.Sort();

            int startIterator = 0;
            int endIterator = 0;

            while (startIterator < times.Count)
            {
                if (startTimes[startIterator] >= endTimes[endIterator])
                { // If no overalp then reducing laptop count to indicate laptop reuse
                    usedLaptops -= 1;
                    endIterator += 1;
                }
                usedLaptops += 1;
                startIterator += 1;
            }
            return usedLaptops;

        }

        private static int LaptopRentalsOptimal1(List<List<int>> times)
        {
            if (times.Count == 0) return 0;
            times.Sort((a, b) => a[0.CompareTo(b[0])]);

            List<List<int>> timesWhenLaptopIsUsed = new List<List<int>>();
            timesWhenLaptopIsUsed.Add(times[0]);
            Heap<List<int>> minHeap = new Heap<List<int>>(timesWhenLaptopIsUsed, (a, b) => { return a[0] < b[0]; });

            for (int idx = 1; idx < times.Count; idx++)
            {
                List<int> currentInterval = times[idx];
                if (minHeap.Peek()[1] <= currentInterval[0])
                { // If no overalp then removing a time to indicate laptop reuse
                    minHeap.Remove();
                }
                minHeap.Insert(currentInterval);
            }
            return timesWhenLaptopIsUsed.Count; //or minHeap.Count;

        }
        //https://www.algoexpert.io/questions/phone-number-mnemonics
        public static Dictionary<char, string[]> DIGIT_LETTERS =
                new Dictionary<char, string[]> {
                { '0', new string[] { "0" } },
                { '1', new string[] { "1" } },
                { '2', new string[] { "a", "b", "c" } },
                { '3', new string[] { "d", "e", "f" } },
                { '4', new string[] { "g", "h", "i" } },
                { '5', new string[] { "j", "k", "l" } },
                { '6', new string[] { "m", "n", "o" } },
                { '7', new string[] { "p", "q", "r", "s" } },
                { '8', new string[] { "t", "u", "v" } },
                { '9', new string[] { "w", "x", "y", "z" } }
        };

        // O(4^n * n) time | O(4^n * n) space - where
        // n is the length of the phone number
        public List<string> PhoneNumberMnemonics(string phoneNumber)
        {
            string[] currentMnemonic = new string[phoneNumber.Length];
            Array.Fill(currentMnemonic, "0");

            List<string> mnemonicsFound = new List<string>();
            PhoneNumberMnemonicsHelper(0, phoneNumber, currentMnemonic, mnemonicsFound);
            return mnemonicsFound;
        }

        public void PhoneNumberMnemonicsHelper(
            int idx,
            string phoneNumber,
            string[] currentMnemonic,
            List<string> mnemonicsFound
        )
        {
            if (idx == phoneNumber.Length)
            {
                string mnemonic = String.Join("", currentMnemonic);
                mnemonicsFound.Add(mnemonic);
            }
            else
            {
                char digit = phoneNumber[idx];
                string[] letters = DIGIT_LETTERS[digit];
                foreach (var letter in letters)
                {
                    currentMnemonic[idx] = letter;
                    PhoneNumberMnemonicsHelper(
                    idx + 1, phoneNumber, currentMnemonic, mnemonicsFound
                    );
                }
            }
        }
        //https://www.algoexpert.io/questions/minimum-waiting-time
        // O(nlogn) time | O(1) space - where n is the number of queries
        public int MinimumWaitingTime(int[] queries)
        {
            Array.Sort(queries);

            int totalWaitingTime = 0;
            for (int idx = 0; idx < queries.Length; idx++)
            {
                int duration = queries[idx];
                int queriesLeft = queries.Length - (idx + 1);
                totalWaitingTime += duration * queriesLeft;
            }

            return totalWaitingTime;
        }
        //https://www.algoexpert.io/questions/min-rewards

        // O(n^2) time | O(n) space - where n is the length of the input array
        public static int MinRewardsNaive(int[] scores)
        {
            int[] rewards = new int[scores.Length];
            Array.Fill(rewards, 1);
            for (int i = 1; i < scores.Length; i++)
            {
                int j = i - 1;
                if (scores[i] > scores[j])
                {
                    rewards[i] = rewards[j] + 1;
                }
                else
                {
                    while (j >= 0 && scores[j] > scores[j + 1])
                    {
                        rewards[j] = Math.Max(rewards[j], rewards[j + 1] + 1);
                        j--;
                    }
                }
            }
            return rewards.Sum();
        }

        // O(n) time | O(n) space - where n is the length of the input array
        public static int MinRewardsOptimal(int[] scores)
        {
            int[] rewards = new int[scores.Length];
            Array.Fill(rewards, 1);
            List<int> localMinIdxs = getLocalMinIdxs(scores);
            foreach (int localMinIdx in localMinIdxs)
            {
                expandFromLocalMinIdx(localMinIdx, scores, rewards);
            }
            return rewards.Sum();
        }

        public static List<int> getLocalMinIdxs(int[] array)
        {
            List<int> localMinIdxs = new List<int>();
            if (array.Length == 1)
            {
                localMinIdxs.Add(0);
                return localMinIdxs;
            }
            for (int i = 0; i < array.Length; i++)
            {
                if (i == 0 && array[i] < array[i + 1]) localMinIdxs.Add(i);
                if (i == array.Length - 1 && array[i] < array[i - 1]) localMinIdxs.Add(i);
                if (i == 0 || i == array.Length - 1) continue;
                if (array[i] < array[i + 1] && array[i] < array[i - 1])
                    localMinIdxs.Add(i);
            }
            return localMinIdxs;
        }

        public static void expandFromLocalMinIdx(
          int localMinIdx, int[] scores, int[] rewards
        )
        {
            int leftIdx = localMinIdx - 1;
            while (leftIdx >= 0 && scores[leftIdx] > scores[leftIdx + 1])
            {
                rewards[leftIdx] = Math.Max(rewards[leftIdx], rewards[leftIdx + 1] + 1);
                leftIdx--;
            }
            int rightIdx = localMinIdx + 1;
            while (rightIdx < scores.Length && scores[rightIdx] > scores[rightIdx - 1]
            )
            {
                rewards[rightIdx] = rewards[rightIdx - 1] + 1;
                rightIdx++;
            }
        }

        // O(n) time | O(n) space - where n is the length of the input array
        public static int MinRewardsOptimal2(int[] scores)
        {
            int[] rewards = new int[scores.Length];
            Array.Fill(rewards, 1);
            for (int i = 1; i < scores.Length; i++)
            {
                if (scores[i] > scores[i - 1]) rewards[i] = rewards[i - 1] + 1;
            }
            for (int i = scores.Length - 2; i >= 0; i--)
            {
                if (scores[i] > scores[i + 1])
                {
                    rewards[i] = Math.Max(rewards[i], rewards[i + 1] + 1);
                }
            }
            return rewards.Sum();
        }

        //https://www.algoexpert.io/questions/caesar-cipher-encryptor
        // O(n) time | O(n) space
        public static string CaesarCypherEncryptor1(string str, int key)
        {
            char[] newLetters = new char[str.Length];
            int newKey = key % 26;
            for (int i = 0; i < str.Length; i++)
            {
                newLetters[i] = getNewLetter(str[i], newKey);
            }
            return new string(newLetters);
        }

        public static char getNewLetter(char letter, int key)
        {
            int newLetterCode = letter + key;
            return newLetterCode <= 122 ? (char)newLetterCode
                                        : (char)(96 + newLetterCode % 122);
        }
        // O(n) time | O(n) space
        public static string CaesarCypherEncryptor2(string str, int key)
        {
            char[] newLetters = new char[str.Length];
            int newKey = key % 26;
            string alphabet = "abcdefghijklmnopqrstuvwxyz";
            for (int i = 0; i < str.Length; i++)
            {
                newLetters[i] = getNewLetter(str[i], newKey, alphabet);
            }
            return new string(newLetters);
        }

        public static char getNewLetter(char letter, int key, string alphabet)
        {
            int newLetterCode = alphabet.IndexOf(letter) + key;
            return alphabet[newLetterCode % 26];
        }


        //https://www.algoexpert.io/questions/generate-document
        // O(m * (n + m)) time | O(1) space - where n is the number
        // of characters and m is the length of the document
        public bool GenerateDocumentNaive(string characters, string document)
        {
            for (int idx = 0; idx < document.Length; idx++)
            {
                char character = document[idx];
                int documentFrequency = countcharFrequency(character, document);
                int charactersFrequency = countcharFrequency(character, characters);
                if (documentFrequency > charactersFrequency)
                {
                    return false;
                }
            }

            return true;
        }

        public int countcharFrequency(char character, string target)
        {
            int frequency = 0;
            for (int idx = 0; idx < target.Length; idx++)
            {
                char c = target[idx];
                if (c == character)
                {
                    frequency += 1;
                }
            }

            return frequency;
        }

        // O(c * (n + m)) time | O(c) space - where n is the number of characters, m
        // is the length of the document, and c is the number of unique characters in
        // the document
        public bool GenerateDocumentOptimal1(string characters, string document)
        {
            HashSet<char> alreadyCounted = new HashSet<char>();

            for (int idx = 0; idx < document.Length; idx++)
            {
                char character = document[idx];
                if (alreadyCounted.Contains(character))
                {
                    continue;
                }

                int documentFrequency = countcharFrequency(character, document);
                int charactersFrequency = countcharFrequency(character, characters);
                if (documentFrequency > charactersFrequency)
                {
                    return false;
                }

                alreadyCounted.Add(character);
            }

            return true;
        }

        // O(n + m) time | O(c) space - where n is the number of characters, m is
        // the length of the document, and c is the number of unique characters in the
        // characters string
        public bool GenerateDocumentOptimal2(string characters, string document)
        {
            Dictionary<char, int> characterCounts = new Dictionary<char, int>();

            for (int idx = 0; idx < characters.Length; idx++)
            {
                char character = characters[idx];
                characterCounts[character] =
                  characterCounts.GetValueOrDefault(character, 0) + 1;
            }

            for (int idx = 0; idx < document.Length; idx++)
            {
                char character = document[idx];
                if (!characterCounts.ContainsKey(character) || characterCounts[character] == 0)
                {
                    return false;
                }

                characterCounts[character] = characterCounts[character] - 1;
            }

            return true;
        }
        //https://www.algoexpert.io/questions/valid-starting-city

        // O(n^2) time | O(1) space - where n is the number of cities
        public int ValidStartingCityNaive(int[] distances, int[] fuel, int mpg)
        {
            int numberOfCities = distances.Length;

            for (int startCityIdx = 0; startCityIdx < numberOfCities; startCityIdx++)
            {
                int milesRemaining = 0;

                for (int currentCityIdx = startCityIdx;
                     currentCityIdx < startCityIdx + numberOfCities;
                     currentCityIdx++)
                {
                    if (milesRemaining < 0)
                    {
                        continue;
                    }

                    int currentCityIdxRotated = currentCityIdx % numberOfCities;

                    int fuelFromCurrentCity = fuel[currentCityIdxRotated];
                    int distanceToNextCity = distances[currentCityIdxRotated];
                    milesRemaining += fuelFromCurrentCity * mpg - distanceToNextCity;
                }

                if (milesRemaining >= 0)
                {
                    return startCityIdx;
                }
            }

            // This line should never be reached if the inputs are correct.
            return -1;
        }

        // O(n) time | O(1) space - where n is the number of cities
        public int ValidStartingCityOptimal(int[] distances, int[] fuel, int mpg)
        {
            int numberOfCities = distances.Length;
            int milesRemaining = 0;

            int indexOfStartingCityCandidate = 0;
            int milesRemainingAtStartingCityCandidate = 0;

            for (int cityIdx = 1; cityIdx < numberOfCities; cityIdx++)
            {
                int distanceFromPreviousCity = distances[cityIdx - 1];
                int fuelFromPreviousCity = fuel[cityIdx - 1];
                milesRemaining += fuelFromPreviousCity * mpg - distanceFromPreviousCity;

                if (milesRemaining < milesRemainingAtStartingCityCandidate)
                {
                    milesRemainingAtStartingCityCandidate = milesRemaining;
                    indexOfStartingCityCandidate = cityIdx;
                }
            }

            return indexOfStartingCityCandidate;

        }
        //https://www.algoexpert.io/questions/count-squares
        // O(n^2) time | O(n) space - where n is the number of points
        public int CountSquares(int[][] points)
        {
            HashSet<string> pointsSet = new HashSet<string>();
            foreach (var point in points)
            {
                pointsSet.Add(pointTostring(point));
            }

            int count = 0;
            foreach (var pointA in points)
            {
                foreach (var pointB in points)
                {
                    if (pointA == pointB)
                    {
                        continue;
                    }

                    double[] midpoint = new double[] {
          (pointA[0] + pointB[0]) / 2.0, (pointA[1] + pointB[1]) / 2.0
        };
                    double xDistanceFromMid = pointA[0] - midpoint[0];
                    double yDistanceFromMid = pointA[1] - midpoint[1];

                    double[] pointC = new double[] {
          midpoint[0] + yDistanceFromMid, midpoint[1] - xDistanceFromMid
        };
                    double[] pointD = new double[] {
          midpoint[0] - yDistanceFromMid, midpoint[1] + xDistanceFromMid
        };

                    if (pointsSet.Contains(dbPointTostring(pointC)) && pointsSet.Contains(dbPointTostring(pointD)))
                    {
                        count++;
                    }
                }
            }
            return count / 4;
        }

        private string pointTostring(int[] point)
        {
            return point[0] + "," + point[1];
        }

        private string dbPointTostring(double[] point)
        {
            if (point[0] % 1 == 0 && point[1] % 1 == 0)
            {
                return (int)point[0] + "," + (int)point[1];
            }
            return point[0] + "," + point[1];
        }

        //https://www.algoexpert.io/questions/largest-rectangle-under-skyline

        //1. O(n^2) time | O(1) space - where n is the number of buildings
        public int LargestRectangleUnderSkylineNaive(List<int> buildings)
        {
            int maxArea = 0;
            for (int pillarIdx = 0; pillarIdx < buildings.Count; pillarIdx++)
            {
                int currentHeight = buildings[pillarIdx];

                int furthestLeft = pillarIdx;
                while (furthestLeft > 0 && buildings[furthestLeft - 1] >= currentHeight)
                {
                    furthestLeft -= 1;
                }

                int furthestRight = pillarIdx;
                while (furthestRight < buildings.Count - 1 &&
                       buildings[furthestRight + 1] >= currentHeight)
                {
                    furthestRight += 1;
                }

                int areaWithCurrentBuilding =
                  (furthestRight - furthestLeft + 1) * currentHeight;
                maxArea = Math.Max(areaWithCurrentBuilding, maxArea);
            }

            return maxArea;
        }


        //2. O(n) time | O(n) space - where n is the number of buildings
        public int LargestRectangleUnderSkylineOptimal(List<int> buildings)
        {
            Stack<int> pillarIndices = new Stack<int>();
            int maxArea = 0;

            List<int> extendedBuildings = new List<int>(buildings);
            extendedBuildings.Add(0);
            for (int idx = 0; idx < extendedBuildings.Count; idx++)
            {
                int height = extendedBuildings[idx];
                while (pillarIndices.Count != 0 &&
                       extendedBuildings[pillarIndices.Peek()] >= height)
                {
                    int pillarHeight = extendedBuildings[pillarIndices.Pop()];
                    int width =
                      (pillarIndices.Count == 0) ? idx : idx - pillarIndices.Peek() - 1;
                    maxArea = Math.Max(width * pillarHeight, maxArea);
                }

                pillarIndices.Push(idx);
            }

            return maxArea;
        }
        //https://www.algoexpert.io/questions/apartment-hunting

        //1. O(b^2*r) time | O(b) space - where b is the number of blocks and r is the
        // number of requirements
        public static int ApartmentHuntingNaive(
          List<Dictionary<string, bool>> blocks, string[] reqs
        )
        {
            int[] maxDistancesAtBlocks = new int[blocks.Count];
            Array.Fill(maxDistancesAtBlocks, Int32.MinValue);

            for (int i = 0; i < blocks.Count; i++)
            {
                foreach (string req in reqs)
                {
                    int closestReqDistance = Int32.MaxValue;
                    for (int j = 0; j < blocks.Count; j++)
                    {
                        if (blocks[j][req])
                        {
                            closestReqDistance =
                              Math.Min(closestReqDistance, distanceBetween(i, j));
                        }
                    }
                    maxDistancesAtBlocks[i] =
                      Math.Max(maxDistancesAtBlocks[i], closestReqDistance);
                }
            }
            return getIdxAtMinValue(maxDistancesAtBlocks);
        }

        public static int getIdxAtMinValue(int[] array)
        {
            int idxAtMinValue = 0;
            int minValue = Int32.MaxValue;
            for (int i = 0; i < array.Length; i++)
            {
                int currentValue = array[i];
                if (currentValue < minValue)
                {
                    minValue = currentValue;
                    idxAtMinValue = i;
                }
            }
            return idxAtMinValue;
        }

        public static int distanceBetween(int a, int b)
        {
            return Math.Abs(a - b);
        }

        //2. O(br) time | O(br) space - where b is the number of blocks and r is the
        // number of requirements
        public static int ApartmentHunting(
          List<Dictionary<string, bool>> blocks, string[] reqs
        )
        {
            int[][] minDistancesFromBlocks = new int[reqs.Length][];
            for (int i = 0; i < reqs.Length; i++)
            {
                minDistancesFromBlocks[i] = getMinDistances(blocks, reqs[i]);
            }
            int[] maxDistancesAtBlocks =
              getMaxDistancesAtBlocks(blocks, minDistancesFromBlocks);
            return getIdxAtMinValue(maxDistancesAtBlocks);
        }

        public static int[] getMinDistances(
          List<Dictionary<string, bool>> blocks, string req
        )
        {
            int[] minDistances = new int[blocks.Count];
            int closestReqIdx = Int32.MaxValue;
            for (int i = 0; i < blocks.Count; i++)
            {
                if (blocks[i][req]) closestReqIdx = i;
                minDistances[i] = distanceBetween(i, closestReqIdx);
            }
            for (int i = blocks.Count - 1; i >= 0; i--)
            {
                if (blocks[i][req]) closestReqIdx = i;
                minDistances[i] =
                  Math.Min(minDistances[i], distanceBetween(i, closestReqIdx));
            }
            return minDistances;
        }
        public static int[] getMaxDistancesAtBlocks(
   List<Dictionary<string, bool>> blocks, int[][] minDistancesFromBlocks
 )
        {
            int[] maxDistancesAtBlocks = new int[blocks.Count];
            for (int i = 0; i < blocks.Count; i++)
            {
                int[] minDistancesAtBlock = new int[minDistancesFromBlocks.Length];
                for (int j = 0; j < minDistancesFromBlocks.Length; j++)
                {
                    minDistancesAtBlock[j] = minDistancesFromBlocks[j][i];
                }
                maxDistancesAtBlocks[i] = arrayMax(minDistancesAtBlock);
            }
            return maxDistancesAtBlocks;
        }

        public static int arrayMax(int[] array)
        {
            int max = array[0];
            foreach (int a in array)
            {
                if (a > max)
                {
                    max = a;
                }
            }
            return max;
        }

        //https://www.algoexpert.io/questions/airport-connections

        // O(a * (a + r) + a + r + alog(a)) time | O(a + r) space - where a is the
        // number of airports and r is the number of routes
        public static int AirportConnections(
          List<string> airports, List<List<string>> routes, string startingAirport
        )
        {
            Dictionary<string, AirportNode> airportGraph =
              createAirportGraph(airports, routes);
            List<AirportNode> unreachableAirportNodes =
              getUnreachableAirportNodes(airportGraph, airports, startingAirport);
            markUnreachableConnections(airportGraph, unreachableAirportNodes);
            return getMinNumberOfNewConnections(airportGraph, unreachableAirportNodes);
        }

        // O(a + r) time | O(a + r) space
        public static Dictionary<string, AirportNode> createAirportGraph(
          List<string> airports, List<List<string>> routes
        )
        {
            Dictionary<string, AirportNode> airportGraph =
              new Dictionary<string, AirportNode>();
            foreach (string airport in airports)
            {
                airportGraph.Add(airport, new AirportNode(airport));
            }
            foreach (List<string> route in routes)
            {
                string airport = route[0];
                string connection = route[1];
                airportGraph[airport].connections.Add(connection);
            }
            return airportGraph;
        }

        // O(a + r) time | O(a) space
        public static List<AirportNode> getUnreachableAirportNodes(
          Dictionary<string, AirportNode> airportGraph,
          List<string> airports,
          string startingAirport
        )
        {
            HashSet<string> visitedAirports = new HashSet<string>();
            depthFirstTraverseAirports(airportGraph, startingAirport, visitedAirports);

            List<AirportNode> unreachableAirportNodes = new List<AirportNode>();
            foreach (string airport in airports)
            {
                if (visitedAirports.Contains(airport)) continue;
                AirportNode airportNode = airportGraph[airport];
                airportNode.isReachable = false;
                unreachableAirportNodes.Add(airportNode);
            }
            return unreachableAirportNodes;
        }

        public static void depthFirstTraverseAirports(
          Dictionary<string, AirportNode> airportGraph,
          string airport,
          HashSet<string> visitedAirports
        )
        {
            if (visitedAirports.Contains(airport)) return;
            visitedAirports.Add(airport);
            List<string> connections = airportGraph[airport].connections;
            foreach (string connection in connections)
            {
                depthFirstTraverseAirports(airportGraph, connection, visitedAirports);
            }
        }
        // O(a * (a + r)) time | O(a) space
        public static void markUnreachableConnections(
          Dictionary<string, AirportNode> airportGraph,
          List<AirportNode> unreachableAirportNodes
        )
        {
            foreach (AirportNode airportNode in unreachableAirportNodes)
            {
                string airport = airportNode.airport;
                List<string> unreachableConnections = new List<string>();
                HashSet<string> visitedAirports = new HashSet<string>();
                depthFirstAddUnreachableConnections(
                  airportGraph, airport, unreachableConnections, visitedAirports
                );
                airportNode.unreachableConnections = unreachableConnections;
            }
        }

        public static void depthFirstAddUnreachableConnections(
          Dictionary<string, AirportNode> airportGraph,
          string airport,
          List<string> unreachableConnections,
          HashSet<string> visitedAirports
        )
        {
            if (airportGraph[airport].isReachable) return;
            if (visitedAirports.Contains(airport)) return;
            visitedAirports.Add(airport);
            unreachableConnections.Add(airport);
            List<string> connections = airportGraph[airport].connections;
            foreach (string connection in connections)
            {
                depthFirstAddUnreachableConnections(
                  airportGraph, connection, unreachableConnections, visitedAirports
                );
            }
        }
        // O(alog(a) + a + r) time | O(1) space
        public static int getMinNumberOfNewConnections(
          Dictionary<string, AirportNode> airportGraph,
          List<AirportNode> unreachableAirportNodes
        )
        {
            unreachableAirportNodes.Sort(
              (a1, a2) =>
                a2.unreachableConnections.Count - a1.unreachableConnections.Count
            );
            int numberOfNewConnections = 0;
            foreach (AirportNode airportNode in unreachableAirportNodes)
            {
                if (airportNode.isReachable) continue;
                numberOfNewConnections++;
                foreach (string connection in airportNode.unreachableConnections)
                {
                    airportGraph[connection].isReachable = true;
                }
            }
            return numberOfNewConnections;
        }

        public class AirportNode
        {
            public string airport;
            public List<string> connections;
            public bool isReachable;
            public List<string> unreachableConnections;

            public AirportNode(string airport)
            {
                this.airport = airport;
                connections = new List<string>();
                isReachable = true;
                unreachableConnections = new List<string>();
            }
        }
        //https://www.algoexpert.io/questions/optimalAssemblyLine
        // O(n * log(m)) time | O(1) space - where n is the length of stepDurations,
        // and m is the sum of all values in stepDurations
        public int OptimalAssemblyLine(int[] stepDurations, int numStations)
        {
            int left = Int32.MinValue;
            int right = 0;
            int maxStationDuration = Int32.MaxValue;

            foreach (var stepDuration in stepDurations)
            {
                left = Math.Max(left, stepDuration);
                right += stepDuration;
            }

            while (left <= right)
            {
                int potentialMaxStationDuration = (left + right) / 2;

                if (isPotentialSolution(
                      stepDurations, numStations, potentialMaxStationDuration
                    ))
                {
                    maxStationDuration = potentialMaxStationDuration;
                    right = potentialMaxStationDuration - 1;
                }
                else
                {
                    left = potentialMaxStationDuration + 1;
                }
            }
            return maxStationDuration;
        }

        static bool isPotentialSolution(
          int[] stepDurations, int numStations, int potentialMaxStationDuration
        )
        {
            int stationsRequired = 1;
            int currentDuration = 0;

            foreach (var stepDuration in stepDurations)
            {
                if (currentDuration + stepDuration > potentialMaxStationDuration)
                {
                    stationsRequired++;
                    currentDuration = stepDuration;
                }
                else
                {
                    currentDuration += stepDuration;
                }
            }

            return stationsRequired <= numStations;
        }

        //638: Shopping Offers
        //https://youtu.be/iBwv-2IG-DQ?t=330
        //https://leetcode.com/problems/shopping-offers/

        public static int LowerPriceToPayForGivenNeedsAndOffers(List<int> price, List<List<int>> offers, List<int> needs)
        {
            /*
             * Discard any offer that has a higher price than buying individually
             * always use offers first
             * remaining units for each iterm can be filled individually
             */
            int res = 0;

            //1.Using Recursion
            res = LowerPriceToPayForGivenNeedsAndOffersRec(price, offers, needs);

            //2.Using Recursion with memoization 
            res = LowerPriceToPayForGivenNeedsAndOffersRecOptimalDP(price, offers, needs, new Dictionary<List<int>, int>());
            return res;
        }

        private static int LowerPriceToPayForGivenNeedsAndOffersRecOptimalDP(List<int> price, List<List<int>> offers, List<int> needs, Dictionary<List<int>, int> map)
        {
            if (map.ContainsKey(needs))
                return map[needs];

            int j = 0, res = TotalPriceForNeedsWithoutOffers(price, needs);
            foreach (List<int> offer in offers)
            {
                List<int> cloNeeds = new List<int>(needs);

                for (j = 0; j < needs.Count; j++)
                {
                    int diff = cloNeeds[j] - offer[j];
                    if (diff < 0) break; //Can't use this offer as need of an item is more

                    cloNeeds[j] = diff;
                }
                if (j == needs.Count)
                {
                    res = Math.Min(res, offer[j] + LowerPriceToPayForGivenNeedsAndOffersRecOptimalDP(price, offers, cloNeeds, map));
                }
            }
            map[needs] = res;
            return res;

        }

        private static int LowerPriceToPayForGivenNeedsAndOffersRec(List<int> price, List<List<int>> offers, List<int> needs)
        {
            int j = 0;
            //Determind cost of buying items per needs array
            int res = TotalPriceForNeedsWithoutOffers(price, needs);

            foreach (List<int> offer in offers)
            {
                List<int> cloNeeds = new List<int>(needs);

                for (j = 0; j < needs.Count; j++)
                {
                    int diff = cloNeeds[j] - offer[j];
                    if (diff < 0) break; //Can't use this offer as need of an item is more

                    cloNeeds[j] = diff;
                }
                if (j == needs.Count)
                {
                    res = Math.Min(res, offer[j] + LowerPriceToPayForGivenNeedsAndOffersRec(price, offers, cloNeeds));
                }
            }
            return res;

        }

        private static int TotalPriceForNeedsWithoutOffers(List<int> price, List<int> needs)
        {
            int sum = 0;

            for (int i = 0; i < price.Count; ++i)
            {
                sum += price[i] * needs[i];
            }
            return sum;
        }

        //1101. The Earliest Moment When Everyone Become Friends

        //https://leetcode.com/problems/the-earliest-moment-when-everyone-become-friends	
        public int EarliestAcq(int[][] logs, int n)
        {
            //Time complexity: Sorting the logs in O(nlogn) complexity, the union find with rank and path compression is O(logn).
            //O(n)

            int[] parents = Enumerable.Range(0, n).ToArray();
            int[] rank = new int[n];

            // Finds the parent of a vertex
            int Find(int vertex)
            {
                if (vertex == parents[vertex])
                {
                    return vertex;
                }
                int parent = Find(parents[vertex]);
                parents[vertex] = parent;
                return parent;
            }

            // Creates a union of the sets that verticies 
            // v1 and v2 belong to
            int Union(int v1, int v2)
            {
                int p1 = Find(v1);
                int p2 = Find(v2);
                if (p1 == p2)
                {
                    return 0; // Already friends
                }

                // Different parents, so set new parents based on rank.
                // "Rank" is simillar to the height of a tree
                // from the parent
                if (rank[p1] < rank[p2])
                {
                    parents[p1] = p2;
                }
                else if (rank[p1] > rank[p2])
                {
                    parents[p2] = p1;
                }
                else
                {
                    // Both parents have the same rank, so choose a
                    // parent and increase its rank.
                    parents[p2] = p1;
                    rank[p1]++;
                }

                return 1; // New friends
            }

            int friendCount = 1;
            Array.Sort(logs, (x, y) => x[0].CompareTo(y[0]));
            foreach (var log in logs)
            {
                friendCount += Union(log[1], log[2]);
                if (friendCount == n)
                {
                    return log[0];
                }
            }
            return -1;
        }
        //657. Robot Return to Origin
        //https://leetcode.com/problems/robot-return-to-origin
        /*
            public boolean judgeCircle(String moves) {
        int x = 0, y = 0;
        for (char move: moves.toCharArray()) {
            if (move == 'U') y--;
            else if (move == 'D') y++;
            else if (move == 'L') x--;
            else if (move == 'R') x++;
        }
        return x == 0 && y == 0;
    }
        */

        //134. Gas Station		

        //https://leetcode.com/problems/gas-station
        public int CanCompleteCircuit(int[] gas, int[] cost)
        {   //Time complexity: O(n)
            //Space complexity: O(1)


            if (gas.Sum() < cost.Sum()) return -1; //if this condition is false then we have a solution

            int total = 0;
            int startIndex = 0;
            for (int index = 0; index < gas.Length; index++)
            {
                total += (gas[index] - cost[index]);

                if (total < 0)
                {
                    // when difference is negative, then we have to start again
                    total = 0;
                    startIndex = index + 1;
                }
            }

            return startIndex;
        }
        //605. Can Place Flowers
        //https://leetcode.com/problems/can-place-flowers
        //https://www.youtube.com/watch?v=W6pc-vhh-SA
        public bool CanPlaceFlowers(int[] flowerbed, int n)
        {
            int[] bed = new int[flowerbed.Length + 2];
            // for planting a flower, we have to need 3 adjacent empty places
            // for ensuring this, we add '0' (empty places) into the begging and ending of the given array
            Array.Copy(flowerbed, 0, bed, 1, flowerbed.Length);

            for (int i = 1; i < bed.Length - 1; ++i)
            {
                if (bed[i - 1] == 0 && bed[i] == 0 && bed[i + 1] == 0)
                {
                    bed[i] = 1;
                    --n;
                }
                if (n == 0)
                    return true;
            }
            return n <= 0;
        }
        /*
        721. Accounts Merge
        https://leetcode.com/problems/accounts-merge
        */
        public IList<IList<string>> AccountsMerge(IList<IList<string>> accounts)
        {
            //Time complexity: O(NKlogNK) - Here N is the number of accounts and K is the maximum length of an account.
            //Space complexity: O(NK)


            Dictionary<string, int> dict = new();
            DisjointSet ds = new DisjointSet(accounts.Count);
            for (int i = 0; i < accounts.Count; i++)
            {
                for (int j = 1; j < accounts[i].Count; j++)
                {
                    if (dict.ContainsKey(accounts[i][j]))
                    {
                        ds.UnionBySize(dict[accounts[i][j]], i);
                    }
                    else
                        dict.Add(accounts[i][j], i);
                }
            }
            //Way 1
            List<string>[] mergedMail = new List<string>[accounts.Count];
            for (int i = 0; i < accounts.Count; i++)
                mergedMail[i] = new List<string>();
            foreach (var item in dict)
            {
                int ulp = ds.FindUParent(item.Value);
                mergedMail[ulp].Add(item.Key);
            }
            var ans = new List<IList<string>>();
            for (int i = 0; i < accounts.Count; i++)
            {
                if (mergedMail[i].Count == 0)
                    continue;
                mergedMail[i].Sort(StringComparer.Ordinal);
                var temp = new List<string>();
                temp.Add(accounts[i][0]);
                temp.AddRange(mergedMail[i]);
                ans.Add(temp);
            }
            return ans;

            //Way 2
            // var ans = new Dictionary<int,IList<string>>();
            // var mergedMails = dict.Keys.ToList();
            // mergedMails.Sort(StringComparer.Ordinal);
            // foreach(var item in mergedMails)
            // {
            //     int ulp = ds.FindUParent(dict[item]);
            //     if(ans.ContainsKey(ulp))
            //     ans[ulp].Add(item);
            //     else
            //     ans.Add(ulp, new List<string>{accounts[ulp][0], item});
            // }
            // return ans.Values.ToList();
        }

        /*
        2127. Maximum Employees to Be Invited to a Meeting
        https://leetcode.com/problems/maximum-employees-to-be-invited-to-a-meeting        
        */
        public int MaximumInvitations(int[] favorite)
        {
            //Time complexity: O(N) where N is the total number of employees
            //O(N) where N is the total number of employees

            var graph = new Dictionary<int, List<int>>(); // <fav, empls>
            for (int empl = 0; empl < favorite.Length; empl++)
            {
                var fav = favorite[empl];
                if (!graph.ContainsKey(fav))
                {
                    graph[fav] = new List<int>();
                }

                graph[fav].Add(empl);
            }

            // Find all valid 2 person seating arrangement
            var visited = new HashSet<int>();
            var maxInvites = 0;
            var twoManSeatings = 0;
            for (int empl = 0; empl < favorite.Length; empl++)
            {
                if (visited.Contains(empl))
                {
                    continue;
                }

                var fav = favorite[empl];
                if (favorite[fav] == empl)
                {
                    // extend the seating arrangement 
                    visited.Add(fav);
                    visited.Add(empl);
                    maxInvites += ExtendSeating(empl, visited, graph);
                    maxInvites += ExtendSeating(fav, visited, graph);
                }
            }

            for (int empl = 0; empl < favorite.Length; empl++)
            {
                if (!visited.Contains(empl))
                {
                    maxInvites = Math.Max(
                        maxInvites,
                        Dfs(empl, favorite, visited, new List<int>(), new HashSet<int>())
                    );
                }
            }

            return maxInvites;

        }

        public int ExtendSeating(int empl, HashSet<int> visited, Dictionary<int, List<int>> graph)
        {
            var queue = new Queue<int>();
            queue.Enqueue(empl);
            int seats = 0;

            while (queue.Count > 0)
            {
                seats += 1;
                int count = queue.Count;
                for (int sz = 0; sz < count; sz++)
                {
                    var fav = queue.Dequeue();
                    if (!graph.ContainsKey(fav))
                    {
                        continue;
                    }

                    foreach (var nextEmpl in graph[fav])
                    {
                        if (!visited.Contains(nextEmpl))
                        {
                            visited.Add(nextEmpl);
                            queue.Enqueue(nextEmpl);
                        }
                    }
                }
            }

            return seats;
        }

        public int Dfs(int empl, int[] favorite, HashSet<int> visited, List<int> seating, HashSet<int> currVisits)
        {
            if (visited.Contains(empl))
            {
                if (!currVisits.Contains(empl))
                {
                    return 0;
                }

                int index = 0;
                for (index = 0; index < seating.Count; index++)
                {
                    if (seating[index] == empl)
                    {
                        break;
                    }
                }

                return seating.Count - index;
            }

            visited.Add(empl);
            currVisits.Add(empl);
            seating.Add(empl);
            return Dfs(favorite[empl], favorite, visited, seating, currVisits);
        }

        /*
        2050. Parallel Courses III
        https://leetcode.com/problems/parallel-courses-iii       
        */

        //Dependency
        //1. Topological sorting
        public int MinimumTime(int n, int[][] relations, int[] time)
        {
            /*
            •	Time complexity: O(n+e) It costs O(e) to build graph and O(n) to initialize maxTime, queue, and indegree.
            •	Space complexity: O(n+e) graph takes O(n+e) space, the queue can take up to O(n) space, maxTime and indegree both take O(n) space
            */

            // Build the graph and calculate indegrees
            Dictionary<int, List<int>> graph = new Dictionary<int, List<int>>();
            int[] indegrees = new int[n];
            int[] maxTime = new int[n];

            foreach (var relation in relations)
            {
                int prevCourse = relation[0] - 1;
                int nextCourse = relation[1] - 1;

                if (!graph.ContainsKey(prevCourse))
                    graph[prevCourse] = new List<int>();

                graph[prevCourse].Add(nextCourse);
                indegrees[nextCourse]++;
            }

            // Initialize a queue for topological sorting
            Queue<int> queue = new Queue<int>();

            // Add courses with no prerequisites to the queue
            for (int i = 0; i < n; i++)
            {
                if (indegrees[i] == 0)
                {
                    queue.Enqueue(i);
                    maxTime[i] = time[i];
                }
            }

            // Perform topological sorting
            while (queue.Count > 0)
            {
                int currentCourse = queue.Dequeue();

                if (!graph.ContainsKey(currentCourse))
                    continue;

                foreach (var nextCourse in graph[currentCourse])
                {
                    indegrees[nextCourse]--;
                    maxTime[nextCourse] = Math.Max(maxTime[nextCourse], maxTime[currentCourse] + time[nextCourse]);

                    if (indegrees[nextCourse] == 0)
                        queue.Enqueue(nextCourse);
                }
            }

            // Find the maximum time taken to complete all courses
            int minTotalTime = 0;
            foreach (int timeTaken in maxTime)
            {
                minTotalTime = Math.Max(minTotalTime, timeTaken);
            }

            return minTotalTime;
        }
        /*
        630. Course Schedule III
        https://leetcode.com/problems/course-schedule-iii/description/

        */
        public static int ScheduleCourse(int[][] courses)
        {
            int maxCourses = 0;
            //1. Brute Force
            //T:O((n+1)!) | S:O(n)

            //2: Recursion with Memoization
            /*
            Time complexity : O(n∗d). memo array of size nxd is filled once. Here, n refers to the number of courses in the given courses array and d refers to the maximum value of the end day from all the end days in the courses array.
            Space complexity : O(n∗d). memo array of size nxd is used.
            */

            Array.Sort(courses, (a, b) => a[1].CompareTo(b[1]));         // sort courses by last day in ascending order

            //Array.Sort(courses, Comparer<int[]>.Create((left, right) => left[1] - right[1]));

            int?[][] memo = new int?[courses.Length][];
            for (int j = 0; j < memo.Length; j++)
            {
                memo[j] = new int?[courses[courses.Length - 1][1] + 1];
            }
            maxCourses = ScheduleCourseRecDP(courses, 0, 0, memo);

            //3: Iterative
            /*
            Time complexity : O(n^2). We iterate over the count array of size n once. For every element currently considered, we could scan backwards till the first element, giving O(n^2) complexity. 
                            Sorting the count array takes O(nlogn) time for count array.
            Space complexity : O(1). Constant extra space is used.
            */
            maxCourses = ScheduleCourseIterative(courses);

            //4: Optimized Iterative
            /*
            Time complexity : O(n∗count). We iterate over a total of n elements of the courses array. For every element, we can traverse backwards upto at most count(final value) number of elements.
            Space complexity : O(1). Constant extra space is used.
            */
            maxCourses = ScheduleCourseIterativeOptimal(courses);

            //5: Priority List
            /*
            Time complexity : O(nlogn). At most n elements are added to the queue. Adding each element is followed by heapification, which takes O(logn) time.
            Space complexity : O(n). The queue containing the durations of the courses taken can have at most n elements
            */
            maxCourses = ScheduleCoursePriorityQueue1(courses);


            return maxCourses;
        }


        private static int ScheduleCoursePriorityQueue1(int[][] courses)
        {

            PriorityQueue<int, int> priorityQueue = new();  //Stores our processed courses in a min heap structure

            int time = 0; //Track the total # of days we have taken
            foreach (int[] course in courses)
            {
                (int duration, int lastDay) = (course[0], course[1]);
                time += duration;
                priorityQueue.Enqueue(duration, -duration);  //Lowest priority dequeued first -> Set all durations to negative so longest are lowest

                if (time > lastDay) //If our total time takes us over the last day
                {
                    time -= priorityQueue.Dequeue(); //Remove course taking the longest time
                }
            }
            //At this point, our pq has the maximum # of courses we can fit at once.
            return priorityQueue.Count;


        }
        public int ScheduleCoursePriorityQueue2(int[][] courses)
        {
            PriorityQueue<Tuple<int, int>, int> pq = new(Comparer<int>.Create((a, b) => a.CompareTo(b)));
            for (int i = 0; i < courses.Length; i++)
                pq.Enqueue(Tuple.Create(courses[i][0], courses[i][1]), courses[i][1]);

            int curDay = 0;
            PriorityQueue<int, int> biggerDuration = new(Comparer<int>.Create((a, b) => b.CompareTo(a)));
            while (pq.Count > 0)
            {
                var pair = pq.Dequeue();
                if (pair.Item1 + curDay <= pair.Item2)
                {
                    biggerDuration.Enqueue(pair.Item1, pair.Item1);
                    curDay += pair.Item1;
                    continue;
                }
                // exceed & make swap if legal
                if (biggerDuration.Count > 0 && biggerDuration.Peek() > pair.Item1)
                {
                    curDay = curDay - biggerDuration.Peek() + pair.Item1;
                    biggerDuration.Dequeue();
                    biggerDuration.Enqueue(pair.Item1, pair.Item1);
                }
            }
            return biggerDuration.Count;
        }

        public class MinIntComparer : IComparer<int>
        {
            public int Compare(int x, int y) => x < y ? -1 : (x > y ? 1 : 0);
        }


        private static int ScheduleCourseIterativeOptimal(int[][] courses)
        {
            int time = 0, count = 0;
            for (int i = 0; i < courses.Length; i++)
            {
                if (time + courses[i][0] <= courses[i][1])
                {
                    time += courses[i][0];
                    courses[count++] = courses[i];

                }
                else
                {
                    int max_i = i;
                    for (int j = 0; j < count; j++)
                    {
                        if (courses[j][0] > courses[max_i][0])
                            max_i = j;
                    }
                    if (courses[max_i][0] > courses[i][0])
                    {
                        time += courses[i][0] - courses[max_i][0];
                        courses[max_i] = courses[i];

                    }
                }

            }
            return count;

        }

        private static int ScheduleCourseIterative(int[][] courses)
        {
            int time = 0, count = 0;
            for (int i = 0; i < courses.Length; i++)
            {
                if (time + courses[i][0] <= courses[i][1])
                {
                    time += courses[i][0];
                    count++;
                }
                else
                {
                    int max_i = i;
                    for (int j = 0; j < i; j++)
                    {
                        if (courses[j][0] > courses[max_i][0])
                            max_i = j;
                    }
                    if (courses[max_i][0] > courses[i][0])
                    {
                        time += courses[i][0] - courses[max_i][0];
                    }
                    courses[max_i][0] = -1;
                }

            }
            return count;
        }

        public static int ScheduleCourseRecDP(int[][] courses, int i, int time, int?[][] memo)
        {
            if (i == courses.Length)
                return 0;
            if (memo[i][time] != null)
                return memo[i][time].Value;
            int taken = 0;

            if (time + courses[i][0] <= courses[i][1])
                taken = 1 + ScheduleCourseRecDP(courses, i + 1, time + courses[i][0], memo);

            int notTaken = ScheduleCourseRecDP(courses, i + 1, time, memo);

            memo[i][time] = Math.Max(taken, notTaken);
            return memo[i][time].Value;
        }
        /*
        207. Course Schedule
        https://leetcode.com/problems/course-schedule/description
        //https://www.youtube.com/watch?v=cIBFEhD77b4
        Topological sorting

        •	Time complexity: O(m+n)
                o	Initializing the adj list takes O(m) time as we go through all the edges. The indegree array take O(n) time.
                o	Each queue operation takes O(1) time, and a single node will be pushed once, leading to O(n) operations for n nodes. We iterate over the neighbors of each node that is popped out of the queue iterating over all the edges once. Since there are total of m edges, it would take O(m) time to iterate over the edges.
        •	Space complexity: O(m+n)
                o	The adj arrays takes O(m) space. The indegree array takes O(n) space.
                o	The queue can have no more than n elements in the worst-case scenario. It would take up O(n) space in that case.

        */
        public static bool CanFinishCourses(int numCourses, int[,] preReq)
        {
            if (preReq.Length == 0)
                return true; // no cycle could be formed in empty graph.

            Dictionary<int, GNode> graph = new Dictionary<int, GNode>();

            //Build Graph first - Adjacency List
            for (int i = 0; i < preReq.GetLength(0); i++)
            {
                GNode prevCourse = GetCreatedGNode(graph, preReq[i, 1]);
                GNode nxtCourse = GetCreatedGNode(graph, preReq[i, 0]);

                prevCourse.outNodes.Add(preReq[i, 0]);
                nxtCourse.inDegrees += 1;


            }
            // We start from courses that have no prerequisites/dependencies
            int totalDependencies = preReq.GetLength(0);

            Queue<int> noDepCourseQ = new Queue<int>();
            foreach (int key in graph.Keys)
            {
                if (graph[key].inDegrees == 0)
                    noDepCourseQ.Enqueue(key);
            }
            int removedEdges = 0;

            while (noDepCourseQ.Count() > 0)
            {
                int course = noDepCourseQ.Dequeue();

                foreach (int nxtCourse in graph[course].outNodes)
                {
                    GNode childNode = graph[nxtCourse];
                    childNode.inDegrees -= 1;
                    removedEdges += 1;

                    if (childNode.inDegrees == 0)
                        noDepCourseQ.Enqueue(nxtCourse);
                }

            }
            if (removedEdges != totalDependencies)
                // if there are still some edges left, then there exist some cycles
                // Due to the dead-lock (dependencies), we cannot remove the cyclic edges
                return false;
            else
                return true;


        }
        private class GNode
        {
            public int inDegrees = 0;
            public List<int> outNodes = new List<int>();
        }

        private static GNode GetCreatedGNode(Dictionary<int, GNode> graph, int course)
        {

            if (!graph.ContainsKey(course))
            {
                graph[course] = new GNode();
            }
            return graph[course];
        }
        /*
        210. Course Schedule II
        https://leetcode.com/problems/course-schedule-ii/description
        https://www.youtube.com/watch?v=qe_pQCh09yU&list=RDCMUCnxhETjJtTPs37hOZ7vQ88g&index=2

        topological ordering/sorting

        •	Time Complexity: O(V+E) where V represents the number of vertices and E represents the number of edges
        •	Space Complexity: O(V+E). The in-degree array requires O(V) space

        */
        public static int[] FindCourseOrder(int numCourses, int[,] preReq)
        {
            //bool isPossible = false;
            Dictionary<int, List<int>> adjList = new Dictionary<int, List<int>>();

            int[] inDegree = new int[numCourses];

            int[] topologicalOrder = new int[numCourses];

            // Create the adjacency list representation of the graph
            for (int i = 0; i < preReq.GetLength(0); i++)
            {
                int dest = preReq[i, 0];
                int src = preReq[i, 1];

                if (adjList.ContainsKey(src))
                    adjList[src] = new List<int>();

                adjList[src].Add(dest);

                // Record in-degree of each vertex
                inDegree[dest] += 1;
            }
            if (IsCycleExists(adjList, numCourses))
                return new int[0];

            //***DFS************
            Stack<int> stack = new Stack<int>(numCourses);
            bool[] visited = new bool[numCourses];

            //Apply DFS and and find topological sort
            for (int i = 0; i < numCourses; i++)
                if (!visited[i])
                    FindCourseOrderDfs(adjList, i, visited, stack);

            int s = 0;
            while (stack.Count > 0)
            {
                topologicalOrder[s++] = stack.Peek();
                stack.Pop();
            }
            //***BFS************
            // Add all vertices with 0 in-degree to the queue
            Queue<int> q = new Queue<int>();
            for (int i = 0; i < numCourses; i++)
                if (inDegree[i] == 0)
                    q.Enqueue(i);

            int i1 = 0;
            // Process until the Q becomes empty
            while (q.Count > 0)
            {
                int node = q.Dequeue();

                topologicalOrder[i1++] = node;

                if (adjList.ContainsKey(node))
                {
                    foreach (int i2 in adjList[node])
                    {
                        inDegree[i2]--;

                        // If in-degree of a neighbor becomes 0, add it to the Q
                        if (inDegree[i2] == 0)
                            q.Enqueue(i2);
                    }
                }

            }
            // Check to see if topological sort is possible or not.
            if (i1 == numCourses) return topologicalOrder;

            return new int[0];
        }

        private static void FindCourseOrderDfs(Dictionary<int, List<int>> adjList, int v, bool[] visited, Stack<int> stack)
        {
            visited[v] = true;

            for (int i = 0; i < adjList[v].Count; i++)
                if (!visited[adjList[v][i]])
                    FindCourseOrderDfs(adjList, adjList[v][i], visited, stack);

            stack.Push(v);


        }

        private static bool IsCycleExists(Dictionary<int, List<int>> adjList, int numCourses)
        {
            // 0 - Not visited/ 1- visited / 2 -visited & processed
            int[] visited = new int[numCourses];

            for (int i = 0; i < numCourses; i++)
            {
                if (visited[i] == 0)
                    if (HasCycleDfs(adjList, visited, i))
                        return true;
            }
            return false;
        }

        private static bool HasCycleDfs(Dictionary<int, List<int>> adjList, int[] visited, int v)
        {
            if (visited[v] == 1)
                return true;
            if (visited[v] == 2)
                return false;

            visited[v] = 1; // Mark current node as visited

            for (int i = 0; i < adjList[v].Count; i++)
                if (HasCycleDfs(adjList, visited, adjList[v][i]))
                    return true;

            visited[v] = 2; // Mark current node as processed
            return false;

        }

        /*
        1136. Parallel Courses
        https://leetcode.com/problems/parallel-courses

        topological ordering/sorting

        •	Time Complexity: O(V+E) where V represents the number of vertices and E represents the number of edges
        •	Space Complexity: O(V+E). 
        */
        public int MinimumSemesters(int n, int[][] relations)
        {
            // To track our progress and answer
            int answer = 0;
            int coursesCompleted = 0;

            // This array contains the indegree of each course (arrows coming in)
            //------------------------------------------------
            // Using +1 to use the same numbers, 0 is not used
            //------------------------------------------------
            int[] indegree = new int[n + 1];

            // An array of lists to track courses that come after this one
            Dictionary<int, List<int>> graph = new Dictionary<int, List<int>>();
            // Initiate the graph, blank
            for (int i = 0; i <= n; i++)
            {
                graph[i] = new List<int>();
            }

            // Populate the list of next courses for each course
            foreach (int[] relation in relations)
            {
                // Calculate the next course's indegree (because it has a preRequisite: arrow coming in)
                indegree[relation[1]]++;
                // Add course that follows, to its list
                graph[relation[0]].Add(relation[1]);
            }

            // To control the processing of courses per semester
            Queue<int> queue = new Queue<int>();

            // Check all the indegrees, we start with
            for (int i = 1; i <= n; i++)
            {
                // If it is 0 (has no preRequisite - candidate for first semester)
                if (indegree[i] == 0)
                {
                    // Add to the queue of courses to take on this semester
                    queue.Enqueue(i);
                }
            }

            // Process all of the courses in this semester
            while (queue.Count > 0)
            {
                // Get the amount of courses in this semester, so we know when to stop
                int coursesInSemester = queue.Count;

                // Loop over them
                for (int i = 0; i < coursesInSemester; i++)
                {
                    // Take the current course
                    int currentCourse = queue.Dequeue();
                    // Mark as completed
                    coursesCompleted++;

                    // For each course that comes after this one
                    foreach (int course in graph[currentCourse])
                    {
                        // Reduce its indegree, as we have already completed one preReq
                        indegree[course]--;

                        // If no additional pre-reqs, add to next semester (queue)
                        if (indegree[course] == 0)
                        {
                            queue.Enqueue(course);
                        }
                    }
                }

                // Increase the semester count
                answer++;
            }

            // If we completed all courses
            if (coursesCompleted == n)
            {
                return answer;
            }
            else
            {
                return -1;
            }
        }

        /*
        1494. Parallel Courses II	
        https://leetcode.com/problems/parallel-courses-ii/description/

        DP BIT MASKING

        */
        public int MinNumberOfSemesters(int n, int[][] relations, int k)
        {
            // dp[i] : the minimum number of semesters needed to take the courses with the bit set in i
            // the worst case is that in each semester we can only take one course, hence initialise with `n`
            // at the end, the answer would be dp[(1 << n) - 1], i.e. all bits set
            List<int> dp = new List<int>(new int[1 << n]);
            for (int i = 0; i < (1 << n); i++)
            {
                dp[i] = n;
            }
            // if the i-th bit is set in pre[j], 
            // that means we need to take course i in order to take course j
            List<int> pre = new List<int>(new int[n]);

            // build the prerequisites
            foreach (var x in relations)
            {
                // make it 0-based index
                --x[0];
                // set the bit at x[0]-th place
                --x[1];

                pre[x[1]] |= 1 << x[0];
            }
            // base case: 0 semester. 0 course.
            dp[0] = 0;

            // i is a set of courses that we've already studied
            for (int i = 0; i < (1 << n); i++)
            {
                // init can as 0 to record how can courses we can study                
                int can = 0;
                // iterate all courses
                for (int j = 0; j < n; j++)
                {
                    // check if we've studied prerequisite courses
                    if ((pre[j] & i) == pre[j])
                    {
                        // if so, we can study course j
                        can |= (1 << j);
                    }
                }
                // remove those courses that we've already studied
                can &= ~i;

                // enumerate all the bit 1 combinations of `can`
                // i.e. all subsets of a bit representation

                for (int s = can; Convert.ToBoolean(s); s = ((s - 1) & can))
                {
                    // check if we can take __builtin_popcount(s) courses
                    if (System.Numerics.BitOperations.PopCount((uint)s) <= k)
                    {
                        // if so, we combine the previous results (what've studied already)
                        // or we take a new semester
                        dp[i | s] = Math.Min(dp[i | s], dp[i] + 1);
                    }
                }
            }
            // same as dp[(1 << n) - 1]
            return dp[(1 << n) - 1];
        }

        /*
        851. Loud and Rich
        https://leetcode.com/problems/loud-and-rich/description/

        •	Time Complexity: O(N^2), where N is the number of people.
        •	Space Complexity: O(N^2), to keep the graph with N2 edges.

        */
        public int[] LoudAndRich(int[][] richer, int[] quiet)
        {
            int n = quiet.Length;
            List<int>[] graph = new List<int>[n];
            for (int i = 0; i < n; i++)
            {
                graph[i] = new List<int>();
            }

            foreach (var pair in richer)
            {
                graph[pair[1]].Add(pair[0]);
            }

            int[] answer = new int[n];
            Array.Fill(answer, -1);

            for (int i = 0; i < n; i++)
            {
                DFS(i, graph, quiet, answer);
            }

            return answer;
        }

        private int DFS(int person, List<int>[] graph, int[] quiet, int[] answer)
        {
            if (answer[person] != -1)
            {
                return answer[person];
            }

            int minQuietPerson = person;
            foreach (var richerPerson in graph[person])
            {
                int candidate = DFS(richerPerson, graph, quiet, answer);
                if (quiet[candidate] < quiet[minQuietPerson])
                {
                    minQuietPerson = candidate;
                }
            }

            answer[person] = minQuietPerson;
            return minQuietPerson;
        }
        /*
        2115. Find All Possible Recipes from Given Supplies
        https://leetcode.com/problems/find-all-possible-recipes-from-given-supplies/description/


        */
        public List<string> FindAllRecipes(List<string> recipes, List<List<string>> ingredients, List<string> supplies)
        {
            Dictionary<string, List<string>> recipeGraph = new Dictionary<string, List<string>>();
            int recipeCount = recipes.Count;
            HashSet<string> supplySet = new HashSet<string>(supplies); // Store all the supplies in a hash set

            Dictionary<string, int> recipeIndegree = new Dictionary<string, int>(); // To store the indegree of all recipes
            foreach (var recipe in recipes)
            {
                recipeIndegree[recipe] = 0; // Initially set the indegree of all recipes to be 0
            }

            for (int i = 0; i < recipeCount; i++)
            {
                for (int j = 0; j < ingredients[i].Count; j++)
                {
                    if (!supplySet.Contains(ingredients[i][j])) // If the ingredient required to make a recipe is not in supplies
                    {
                        if (!recipeGraph.ContainsKey(ingredients[i][j]))
                        {
                            recipeGraph[ingredients[i][j]] = new List<string>();
                        }
                        recipeGraph[ingredients[i][j]].Add(recipes[i]); // Create a directed edge from that ingredient to recipe
                        recipeIndegree[recipes[i]]++; // Increase the indegree of the recipe
                    }
                }
            }

            // KAHN'S ALGORITHM
            Queue<string> recipeQueue = new Queue<string>();
            foreach (var recipeEntry in recipeIndegree)
            {
                if (recipeEntry.Value == 0)
                {
                    recipeQueue.Enqueue(recipeEntry.Key);
                }
            }

            List<string> result = new List<string>();
            while (recipeQueue.Count > 0)
            {
                string currentRecipe = recipeQueue.Dequeue();
                result.Add(currentRecipe);
                if (recipeGraph.ContainsKey(currentRecipe))
                {
                    foreach (var neighbor in recipeGraph[currentRecipe])
                    {
                        recipeIndegree[neighbor]--;
                        if (recipeIndegree[neighbor] == 0)
                        {
                            recipeQueue.Enqueue(neighbor);
                        }
                    }
                }
            }
            return result;
        }
        /*
        2076. Process Restricted Friend Requests
        https://leetcode.com/problems/process-restricted-friend-requests/description/

        Time Complexity: O(n * m * log(n))   where m = len(requests).
        Space Complexity: O(n) 
        */
        public bool[] FriendRequests(int n, int[][] restrictions, int[][] requests)
        {
            List<bool> result = new();
            DSUArray uf = new DSUArray(n);

            foreach (var req in requests)
            {
                int parX = uf.Find(req[0]);
                int parY = uf.Find(req[1]);

                bool isRestricted = false;

                foreach (var rest in restrictions)
                {
                    int rX = uf.Find(rest[0]);
                    int rY = uf.Find(rest[1]);

                    if ((parX == rX && parY == rY) || (parX == rY && parY == rX))
                    {
                        isRestricted = true;
                        break;
                    }
                }

                result.Add(!isRestricted);

                if (!isRestricted)
                    uf.Union(parX, parY);
            }

            return result.ToArray();
        }
        /*
        489. Robot Room Cleaner
        https://leetcode.com/problems/robot-room-cleaner/

        Time complexity : O(N−M), where N is a number of cells in the room and M is a number of obstacles. We visit each non-obstacle cell once and only once.
                          At each visit, we will check 4 directions around the cell. Therefore, the total number of operations would be 4⋅(N−M).
        Space complexity : O(N−M), where N is a number of cells in the room and M is a number of obstacles.
                          We employed a hashtable to keep track of whether a non-obstacle cell is visited or not.
        */

        public void CleanRoom(Robot robot)
        {
            this.robot = robot;
            Backtrack(0, 0, 0);

        }
        public void Backtrack(int row, int col, int d)
        {
            visited.Add(new Tuple<int, int>(row, col));
            robot.Clean();
            // going clockwise : 0: 'up', 1: 'right', 2: 'down', 3: 'left'
            for (int i = 0; i < 4; ++i)
            {
                int newD = (d + i) % 4;
                int newRow = row + directions[newD][0];
                int newCol = col + directions[newD][1];

                if (!visited.Contains(new Tuple<int, int>(newRow, newCol)) && robot.Move())
                {
                    Backtrack(newRow, newCol, newD);
                    GoBack();
                }
                // turn the robot following chosen direction : clockwise
                robot.TurnRight();
            }
        }
        public void GoBack()
        {
            robot.TurnRight();
            robot.TurnRight();
            robot.Move();
            robot.TurnRight();
            robot.TurnRight();
        }
        // going clockwise : 0: 'up', 1: 'right', 2: 'down', 3: 'left'
        int[][] directions = new int[][] { new int[] { -1, 0 }, new int[] { 0, 1 }, new int[] { 1, 0 }, new int[] { 0, -1 } };
        HashSet<Tuple<int, int>> visited = new HashSet<Tuple<int, int>>();
        Robot robot;

        // This is the robot's control interface.
        // You should not implement it, or speculate about its implementation
        public interface Robot
        {
            // Returns true if the cell in front is open and robot moves into the cell.
            // Returns false if the cell in front is blocked and robot stays in the current cell.
            public bool Move();

            // Robot will stay in the same cell after calling turnLeft/turnRight.
            // Each turn will be 90 degrees.
            public void TurnLeft();
            public void TurnRight();

            // Clean the current cell.
            public void Clean();
        }
        /*
        994. Rotting Oranges	
        https://leetcode.com/problems/rotting-oranges/description
        */
        public int OrangesRotting(int[][] grid)
        {
            //1.Breadth-First Search (BFS)
            /*
            Time Complexity: O(NM), where N×M is the size of the grid. First, we scan the grid to find the initial values for the queue, which would take O(NM) time.
                            Then we run the BFS process on the queue, which in the worst case would enumerate all the cells in the grid once and only once. Therefore, it takes O(NM) time.
                            Thus combining the above two steps, the overall time complexity would be O(NM)+O(NM)=O(NM)
            Space Complexity: O(NM), where N is the size of the grid. 
                              In the worst case, the grid is filled with rotten oranges. As a result, the queue would be initialized with all the cells in the grid.
                              By the way, normally for BFS, the main space complexity lies in the process rather than the initialization. 
                              For instance, for a BFS traversal in a tree, at any given moment, the queue would hold no more than 2 levels of tree nodes. Therefore, the space complexity of BFS traversal in a tree would depend on the width of the input tree.

            */
            int minutesElapsed = 0;
            minutesElapsed = OrangesRottingBFS(grid);

            //2.In-Place Breadth-First Search (BFS) without using Extra space i.e. Queue in this case but it comprmizes(increases) Time Complexity 
            /*
            Time Complexity: O(N^2*M^2)  where N×M is the size of the input grid.
                            In the in-place BFS traversal, for each round of BFS, we would have to iterate through the entire grid.
                            The contamination propagates in 4 different directions. If the orange is well adjacent to each other, the chain of propagation would continue until all the oranges turn rotten.
                            In the worst case, the rotten and the fresh oranges might be arranged in a way that we would have to run the BFS loop over and over again, which could amount to NM/2 times 
                            which is the longest propagation chain that we might have.
                            As a result, the overall time complexity of the in-place BFS algorithm is O(NM*NM/2) =O(N^2*M^2)
            Space Complexity: O(1), the memory usage is constant regardless the size of the input. This is the very point of applying in-place algorithm. Here we trade the time complexity with the space complexity, which is a common scenario in many algorithms.
            */
            minutesElapsed = OrangesRottingInPlaceBFS(grid);

            return minutesElapsed;

        }

        private int OrangesRottingInPlaceBFS(int[][] grid)
        {
            int ROWS = grid.Length, COLS = grid[0].Length;
            int timestamp = 2;
            while (RunRottingProcess(timestamp, grid, ROWS, COLS))
                timestamp++;

            // end of process, to check if there are still fresh oranges left
            foreach (int[] row in grid)
            {
                foreach (int cell in row)
                {
                    // still got a fresh orange left
                    if (cell == 1)
                    {
                        return -1;
                    }
                }
            }


            // return elapsed minutes if no fresh orange left
            return timestamp - 2;

        }

        // run the rotting process, by marking the rotten oranges with the timestamp
        private bool RunRottingProcess(int timestamp, int[][] grid, int rows, int cols)
        {
            int[][] directions = { new int[] { -1, 0 }, new int[] { 0, 1 }, new int[] { 1, 0 }, new int[] { 0, -1 } };
            // flag to indicate if the rotting process should be continued
            bool toBeContinued = false;
            for (int row = 0; row < rows; ++row)
                for (int col = 0; col < cols; ++col)
                    if (grid[row][col] == timestamp)
                        // current contaminated cell
                        foreach (int[] d in directions)
                        {
                            int nRow = row + d[0], nCol = col + d[1];
                            if (nRow >= 0 && nRow < rows && nCol >= 0 && nCol < cols)
                                if (grid[nRow][nCol] == 1)
                                {
                                    // this fresh orange would be contaminated next
                                    grid[nRow][nCol] = timestamp + 1;
                                    toBeContinued = true;
                                }
                        }
            return toBeContinued;
        }

        private static int OrangesRottingBFS(int[][] grid)
        {
            Queue<(int, int)> queue = new Queue<(int, int)>();

            // Step 1). build the initial set of rotten oranges
            int freshOranges = 0;
            int ROWS = grid.Length, COLS = grid[0].Length;

            for (int r = 0; r < ROWS; ++r)
                for (int c = 0; c < COLS; ++c)
                    if (grid[r][c] == 2)
                        queue.Enqueue((r, c));
                    else if (grid[r][c] == 1)
                        freshOranges++;

            // Mark the round / level, _i.e_ the ticker of timestamp
            queue.Enqueue((-1, -1));

            // Step 2). start the rotting process via BFS
            int minutesElapsed = -1;
            int[][] directions = { new int[] { -1, 0 }, new int[] { 0, 1 }, new int[] { 1, 0 }, new int[] { 0, -1 } };

            while (queue.Count > 0)
            {
                (int row, int col) = queue.Dequeue();
                if (row == -1)
                {
                    // We finish one round of processing
                    minutesElapsed++;
                    // to avoid the endless loop
                    if (queue.Count > 0)
                        queue.Enqueue((-1, -1));
                }
                else
                {
                    // this is a rotten orange
                    // then it would contaminate its neighbors
                    foreach (int[] d in directions)
                    {
                        int neighborRow = row + d[0];
                        int neighborCol = col + d[1];
                        if (neighborRow >= 0 && neighborRow < ROWS &&
                            neighborCol >= 0 && neighborCol < COLS)
                        {
                            if (grid[neighborRow][neighborCol] == 1)
                            {
                                // this orange would be contaminated
                                grid[neighborRow][neighborCol] = 2;
                                freshOranges--;
                                // this orange would then contaminate other oranges
                                queue.Enqueue((neighborRow, neighborCol));
                            }
                        }
                    }
                }
            }
            // return elapsed minutes if no fresh orange left
            return freshOranges == 0 ? minutesElapsed : -1;

        }
        /*
        2258. Escape the Spreading Fire 
        https://leetcode.com/problems/escape-the-spreading-fire/description/
        */
        public int MaximumMinutes(int[][] grid)
        {
            //1. BFS + Binary Search (BS)
            int maximumMinutes = MaximumMinutesBFSWithBS(grid);
            //2.BFS 
            maximumMinutes = MaximumMinutesBFS(grid);

            return maximumMinutes;
        }

        private int MaximumMinutesBFSWithBS(int[][] grid)
        {
            throw new NotImplementedException();
        }

        private int MaximumMinutesBFS(int[][] grid)
        {
            int m = grid.Length, n = grid[0].Length;
            int[][] fireDists = new int[m][];
            int[][] personDists = new int[m][];

            for (int i = 0; i < m; i++)
            {
                Array.Fill(fireDists[i], -1);
                Array.Fill(personDists[i], -1);
            }

            personDists[0][0] = 0;
            Queue<int[]> personCells = new Queue<int[]>();
            personCells.Enqueue(new int[] { 0, 0 }); // [row, col]
            Queue<int[]> fireCells = new Queue<int[]>();

            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    if (grid[i][j] == 1)
                    {
                        fireCells.Enqueue(new int[] { i, j });
                        fireDists[i][j] = 0;
                    }
                }
            }

            Travel(grid, personCells, personDists, m, n);
            Travel(grid, fireCells, fireDists, m, n);

            if (personDists[m - 1][n - 1] == -1)
            {
                return -1;
            }

            if (fireDists[m - 1][n - 1] == -1)
            {
                return (int)1e9;
            }

            if (fireDists[m - 1][n - 1] < personDists[m - 1][n - 1])
            {
                return -1;
            }

            int diff = fireDists[m - 1][n - 1] - personDists[m - 1][n - 1];

            // handle special case where there is a path that can wait one sec longer 
            // to meet with the fire  that is coming from a different path and meet at the last cell.
            // There are only two ways to reach last cell from the top or from the left side since last cell is situated at the right-bottom corner.
            if (fireDists[m - 1][n - 2] - personDists[m - 1][n - 2] > diff || fireDists[m - 2][n - 1] - personDists[m - 2][n - 1] > diff)
            {
                return diff;
            }

            return diff - 1;

        }
        private void Travel(int[][] grid, Queue<int[]> cells, int[][] distances, int m, int n)
        {
            int dist = 0;
            while (cells.Count > 0)
            {
                int size = cells.Count;
                for (int i = 0; i < size; i++)
                {
                    int[] cell = cells.Dequeue();
                    foreach (int[] dir in DIRS)
                    {
                        int a = cell[0] + dir[0], b = cell[1] + dir[1];
                        if (a >= 0 && a < m && b >= 0 && b < n && distances[a][b] == -1 && grid[a][b] == 0)
                        {
                            distances[a][b] = dist + 1;
                            cells.Enqueue(new int[] { a, b });
                        }
                    }
                }
                dist++;
            }
        }

        private static readonly int[][] DIRS = { new int[] { 0, 1 }, new int[] { 0, -1 }, new int[] { -1, 0 }, new int[] { 1, 0 } };

        /*
        1386. Cinema Seat Allocation
        https://leetcode.com/problems/cinema-seat-allocation/description/
        Time complexity:O(n)
        Space complexity:O(n)
        */
        public int MaxNumberOfFamilies(int n, int[][] reservedSeats)
        {
            var reservedDictionary = new Dictionary<int, bool[]>();
            foreach (var reservedSeat in reservedSeats)
            {
                var seat = reservedSeat[1];
                if (seat != 1 && seat != 10)
                {
                    var row = reservedSeat[0];
                    if (!reservedDictionary.ContainsKey(row))
                        reservedDictionary[row] = new bool[4];
                    reservedDictionary[row][seat / 2 - 1] = true;
                }
            }
            return reservedDictionary.Values.Count(RoomForOneFamily) + 2 * (n - reservedDictionary.Count());

            bool RoomForOneFamily(bool[] occupied)
            {
                return occupied[1] ? !(occupied[2] || occupied[3]) : !(occupied[2] && occupied[0]);
            }
        }

        /*
        1235. Maximum Profit in Job Scheduling
        https://leetcode.com/problems/maximum-profit-in-job-scheduling/description
        */
        public int JobScheduling(int[] startTime, int[] endTime, int[] profit)
        {
            //0.Naive - using Pair of Loops
            /*
                Time complexity: O(N^2)
            */
            int maxProfit = JobSchedulingNaive(startTime, endTime, profit);

            //1.Top-Down Dynamic Programming + Binary Search
            /*
            Let N be the length of the jobs array.
            Time complexity: O(NlogN) Sorting jobs according to their starting time will take O(NlogN).
                            The time complexity for the recursion (with memoization) is equal to the number of times findMaxProfit is called times the average time of findMaxProfit. 
                            The number of calls to findMaxProfit is 2∗N because each non-memoized call will call findMaxProfit twice.
                            Each memoized call will take O(1) time while for the non-memoized call, we will perform a binary search that takes O(logN) time, hence the time complexity will be O(NlogN+N).
                            The total time complexity is therefore equal to O(NlogN).
            Space complexity: O(N) Storing the starting time, ending time, and profit of each job will take 3⋅N space. Hence the complexity is O(N).
                                The space complexity of the sorting algorithm depends on the implementation of each programming language. 
                                For instance, in Java, the Arrays.sort() for primitives is implemented as a variant of quicksort algorithm whose space complexity is O(logN). In C++ sort() function provided by STL is a hybrid of Quick Sort, Heap Sort and Insertion Sort with the worst-case space complexity of O(logN). 
                                In C#, Array.Sort() method uses the introspective sort (introsort) algorithm as follows:
                                        1.If the partition size is less than or equal to 16 elements, it uses an insertion sort algorithm.
                                        2.If the number of partitions exceeds 2 * LogN, where N is the range of the input array, it uses a Heapsort algorithm.
                                        3.Otherwise, it uses a Quicksort algorithm.
                                Thus the use of inbuilt sort() function adds O(logN) to space complexity.
                                The result for each position will be stored in memo and position can have the values from 0 to N, thus the space required is O(N). 
                                Also, stack space in recursion is equal to the maximum number of active functions. In the scenario where every job is not scheduled, the function call stack will be of size N.
                                The total space complexity is therefore equal to O(N).
            */
            maxProfit = JobSchedulingTopDownDPBS(startTime, endTime, profit);

            //2.Bottom-Up Dynamic Programming + Binary Search
            /*
            Let N be the length of the jobs array.
            Time complexity: O(NlogN) Sorting jobs according to their starting time will take O(NlogN) time.
                             We iterate over all N jobs from right to left and for each job we perform a binary search which takes O(logN), so this step also requires O(NlogN) time.
                             The total time complexity is therefore equal to O(NlogN).

            Space complexity: O(N) Storing the start time, end time, and profit of each job takes 3⋅N space. Hence the complexity is O(N).
            */
            maxProfit = JobSchedulingBottomUpDPBS(startTime, endTime, profit);

            //3.Sorting + Priority Queue
            /*
            Let N be the length of the jobs array.
            Time complexity: O(NlogN) Sorting jobs according to their starting time will take O(NlogN) time.
                             We iterate over all N jobs from right to left and for each job we perform a binary search which takes O(logN), so this step also requires O(NlogN) time.
                             The total time complexity is therefore equal to O(NlogN).

            Space complexity: O(N) Storing the start time, end time, and profit of each job takes 3⋅N space. Hence the complexity is O(N).
                             Each of the N jobs will be pushed into the heap. In the worst-case scenario, when all N jobs end later than the last job starts, the heap will reach size N.
                             The total space complexity is therefore equal to O(N).
            */
            maxProfit = JobSchedulingSortPQ(startTime, endTime, profit);

            return maxProfit;
        }

        private int JobSchedulingNaive(int[] startTime, int[] endTime, int[] profit)
        {
            int n = startTime.Length;
            List<Job> jobs = new List<Job>();

            for (int i = 0; i < n; i++)
            {
                jobs.Add(new Job(startTime[i], endTime[i], profit[i]));
            }

            jobs.Sort((x, y) => x.EndTime.CompareTo(y.EndTime));

            int[] dp = new int[n + 1];

            for (int i = 1; i <= n; i++)
            {
                dp[i] = Math.Max(dp[i - 1], jobs[i - 1].Profit);
                for (int j = i - 1; j > 0; j--)
                {
                    if (jobs[i - 1].StartTime >= jobs[j - 1].EndTime)
                    {
                        dp[i] = Math.Max(dp[i], dp[j] + jobs[i - 1].Profit);
                        break;
                    }
                }
            }

            return dp[n];

        }
        public class Job
        {
            public int StartTime;
            public int EndTime;
            public int Profit;

            public Job(int start, int end, int prof)
            {
                StartTime = start;
                EndTime = end;
                Profit = prof;
            }
        }
        private int JobSchedulingSortPQ(int[] startTime, int[] endTime, int[] profit)
        {
            List<List<int>> jobs = new List<List<int>>();

            // storing job's details into one list 
            // this will help in sorting the jobs while maintaining the other parameters
            int length = profit.Length;
            for (int i = 0; i < length; i++)
            {
                List<int> currentJob = new List<int>
            {
                startTime[i],
                endTime[i],
                profit[i]
            };

                jobs.Add(currentJob);
            }

            jobs.Sort((a, b) => a[0].CompareTo(b[0]));
            return FindMaxProfitSortPQ(jobs);
        }

        private int FindMaxProfitSortPQ(List<List<int>> jobs)
        {
            int jobCount = jobs.Count, maxProfit = 0;
            // min heap having {endTime, profit}
            //TODO: Replace below SortedList with PriorityQueue
            SortedSet<List<int>> priorityQueue = new SortedSet<List<int>>(new TheComparator());

            for (int i = 0; i < jobCount; i++)
            {
                int start = jobs[i][0], end = jobs[i][1], profit = jobs[i][2];

                // keep popping while the heap is not empty and
                // jobs are not conflicting
                // update the value of maxProfit
                while (priorityQueue.Count > 0 && start >= priorityQueue.Min[0])
                {
                    maxProfit = Math.Max(maxProfit, priorityQueue.Min[1]);
                    priorityQueue.Remove(priorityQueue.Min);
                }

                List<int> combinedJob = new List<int>
                {
                    end,
                    profit + maxProfit
                };

                // push the job with combined profit
                // if no non-conflicting job is present maxProfit will be 0
                priorityQueue.Add(combinedJob);
            }

            // update the value of maxProfit by comparing with
            // profit of jobs that exists in the heap
            while (priorityQueue.Count > 0)
            {
                maxProfit = Math.Max(maxProfit, priorityQueue.Min[1]);
                priorityQueue.Remove(priorityQueue.Min);
            }

            return maxProfit;
        }
        public int JobSchedulingSortPQ1(int[] startTime, int[] endTime, int[] profit)
        {
            var jobs = YieldPairs(startTime, endTime, profit).OrderBy(s => s.StartTime);
            var queue = new PriorityQueue<State, int>(); //MinHeap by default in C#
            var maxProfit = 0;
            foreach (var job in jobs)
            {
                while (queue.TryPeek(out var state, out _) && job.StartTime >= state.EndTime)
                {
                    maxProfit = Math.Max(maxProfit, state.Profit);
                    queue.Dequeue();
                }

                var newState = new State(job.EndTime, job.Profit + maxProfit);
                queue.Enqueue(newState, newState.EndTime);
            }

            while (queue.TryDequeue(out var state, out _))
                maxProfit = Math.Max(maxProfit, state.Profit);

            return maxProfit;
        }



        private IEnumerable<Job> YieldPairs(int[] startTime, int[] endTime, int[] profit)
        {
            for (var idx = 0; idx < startTime.Length; idx++)
                yield return new Job(startTime[idx], endTime[idx], profit[idx]);
        }

        internal record State(int EndTime, int Profit);

        class TheComparator : IComparer<List<int>>
        {
            public int Compare(List<int> list1, List<int> list2)
            {
                return list1[0] - list2[0];
            }
        }

        private int JobSchedulingBottomUpDPBS(int[] startTime, int[] endTime, int[] profit)
        {
            List<List<int>> jobs = new List<List<int>>();

            // storing job's details into one list 
            // this will help in sorting the jobs while maintaining the other parameters
            int length = profit.Length;
            for (int i = 0; i < length; i++)
            {
                List<int> currentJob = new List<int>();
                currentJob.Add(startTime[i]);
                currentJob.Add(endTime[i]);
                currentJob.Add(profit[i]);

                jobs.Add(currentJob);
            }

            jobs.Sort((a, b) => a[0].CompareTo(b[0]));

            // binary search will be used in startTime so store it as separate list
            for (int i = 0; i < length; i++)
            {
                startTime[i] = jobs[i][0];
            }

            return FindMaxProfitBottomUpDPBS(jobs, startTime);

        }
        // maximum number of jobs are 50000
        int[] memo = new int[50001];

        private int FindNextJob(int[] startTime, int lastEndingTime)
        {
            int start = 0, end = startTime.Length - 1, nextIndex = startTime.Length;

            while (start <= end)
            {
                int mid = (start + end) / 2;

                if (startTime[mid] >= lastEndingTime)
                {
                    nextIndex = mid;
                    end = mid - 1;
                }
                else
                {
                    start = mid + 1;
                }
            }
            return nextIndex;
        }

        private int FindMaxProfitBottomUpDPBS(List<List<int>> jobs, int[] startTime)
        {
            int length = startTime.Length;

            for (int position = length - 1; position >= 0; position--)
            {
                // it is the profit made by scheduling the current job 
                int currentProfit = 0;

                // nextIndex is the index of next non-conflicting job
                int nextIndex = FindNextJob(startTime, jobs[position][1]);

                // if there is a non-conflicting job possible add its profit
                // else just consider the current job profit
                if (nextIndex != length)
                {
                    currentProfit = jobs[position][2] + memo[nextIndex];
                }
                else
                {
                    currentProfit = jobs[position][2];
                }

                // storing the maximum profit we can achieve by scheduling 
                // non-conflicting jobs from index position to the end of array
                if (position == length - 1)
                {
                    memo[position] = currentProfit;
                }
                else
                {
                    memo[position] = Math.Max(currentProfit, memo[position + 1]);
                }
            }

            return memo[0];
        }


        private int JobSchedulingTopDownDPBS(int[] startTime, int[] endTime, int[] profit)
        {
            List<List<int>> jobs = new List<List<int>>();

            // marking all values to -1 so that we can differentiate 
            // if we have already calculated the answer or not
            Array.Fill(memo, -1);

            // storing job's details into one list 
            // this will help in sorting the jobs while maintaining the other parameters
            int length = profit.Length;
            for (int i = 0; i < length; i++)
            {
                List<int> currJob = new List<int> { startTime[i], endTime[i], profit[i] };
                jobs.Add(currJob);
            }
            jobs.Sort((a, b) => a[0].CompareTo(b[0]));

            // binary search will be used in startTime so store it as separate list
            for (int i = 0; i < length; i++)
            {
                startTime[i] = jobs[i][0];
            }

            return FindMaxProfitTopDownDPBS(jobs, startTime, length, 0);
        }
        private int FindMaxProfitTopDownDPBS(List<List<int>> jobs, int[] startTime, int n, int position)
        {
            // 0 profit if we have already iterated over all the jobs
            if (position == n)
            {
                return 0;
            }

            // return result directly if it's calculated 
            if (memo[position] != -1)
            {
                return memo[position];
            }

            // nextIndex is the index of next non-conflicting job
            int nextIndex = FindNextJob(startTime, jobs[position][1]);

            // find the maximum profit of our two options: skipping or scheduling the current job
            int maxProfit = Math.Max(FindMaxProfitTopDownDPBS(jobs, startTime, n, position + 1),
                            jobs[position][2] + FindMaxProfitTopDownDPBS(jobs, startTime, n, nextIndex));

            // return maximum profit and also store it for future reference (memoization)
            return memo[position] = maxProfit;
        }
        /*
        2008. Maximum Earnings From Taxi
        https://leetcode.com/problems/maximum-earnings-from-taxi/description/

        Knapsack DP        
        Time O(n + klogk), k = rides.length
        Space O(n)

        */
        public long MaxTaxiEarnings(int n, int[][] rides)
        {
            Array.Sort(rides, (a, b) => a[0].CompareTo(b[0]));
            long[] dp = new long[n + 1];
            int j = 0;
            for (int i = 1; i <= n; ++i)
            {
                dp[i] = Math.Max(dp[i], dp[i - 1]);
                while (j < rides.Length && rides[j][0] == i)
                {
                    dp[rides[j][1]] = Math.Max(dp[rides[j][1]], dp[i] + rides[j][1] - rides[j][0] + rides[j][2]);
                    ++j;
                }
            }
            return dp[n];


        }

        /*
        1353. Maximum Number of Events That Can Be Attended
        https://leetcode.com/problems/maximum-number-of-events-that-can-be-attended/description/

        Greedy algorithm + Sort + Priority Queue

        Time complexity: O(nlog(n))
        Space complexity: O(n)
        */
        public int MaxEvents(int[][] events)
        {
            Array.Sort(events, (a, b) => a[0] - b[0]);
            int n = events.Length;

            PriorityQueue<int, int> toAttend = new();
            int i = 0;
            int count = 0;
            int day = 1;

            while (toAttend.Count > 0 || i < n)
            {
                //Step1: Add events until today OR speed up to next event if no events added
                while (i < n && events[i][0] <= day)
                {
                    toAttend.Enqueue(events[i][1], events[i][1]);
                    i++;
                }
                if (toAttend.Count == 0)
                {
                    day = events[i][0];
                    continue;
                }

                //Step 2: Attend the event ends earlier
                if (toAttend.Count > 0 && day <= toAttend.Peek())
                {
                    toAttend.Dequeue();
                    count++;
                }
                day++;

                //Step 3: Remove expired events in the queue
                while (toAttend.Count > 0 && toAttend.Peek() < day)
                {
                    toAttend.Dequeue();
                }
            }

            return count;
        }
        /*
        Time O(d + nlogn), where D is the range of A[i][1]
        Space O(N)
        */
        public int MaxEvents2(int[][] events)
        {
            PriorityQueue<int, int> pq = new PriorityQueue<int, int>();
            Array.Sort(events, (a, b) => a[0].CompareTo(b[0]));
            int i = 0, res = 0, n = events.Length;
            for (int d = 1; d <= 100000; ++d)
            {
                while (pq.Count > 0 && pq.Peek() < d)
                    pq.Dequeue();
                while (i < n && events[i][0] == d)
                {
                    pq.Enqueue(events[i][1], events[i][1]);
                    i++;
                }
                if (pq.Count > 0)
                {
                    pq.Dequeue();
                    ++res;
                }
            }
            return res;
        }
        /*
        1751. Maximum Number of Events That Can Be Attended II
        https://leetcode.com/problems/maximum-number-of-events-that-can-be-attended-ii/description/
        */
        public int MaxValue(int[][] events, int k)
        {
            //0.Top-down Dynamic Programming Without Binary Search (Time Limit Exceed)
            /*
            Let n be the length of the input string s.
            Time complexity: O(n⋅(n⋅k+logn)) Sorting events takes O(nlogn) time. We build a 2D array dp of size O(n×k) as memory. The extra parameter prev_ending_time creates many more states, the value of each state in the dp array is computed once but is visited at most O(n) times.
            Space complexity: O(n⋅k) dp takes O(n×k) space.
            */
            int maxValue = MaxValueNaive(events, k);
            //1.Top-down Dynamic Programming + Binary Search
            /*
            Let n be the length of the input string s.
            Time complexity: O(n⋅k⋅logn) Sorting events takes O(nlogn) time. We build dp, a 2D array of size O(n×k) as memory, equal to the number of possible states. Each state is computed with a binary search over all start times, which takes O(logn).
            Space complexity: O(n⋅k) We build a 2D array of size O(n×k) as memory. In the Python solution, we also create an array with length n, which takes O(n) space.
                             The space complexity of a recursive call depends on the maximum depth of the recursive call stack, which is n+k. As each recursive call either increments cur_index by 1 and/or decrements count by 1. 
                            Therefore, at most O(n+k) levels of recursion will be created, and each level consumes a constant amount of space.
            */
            maxValue = MaxValueTDDPBS(events, k);

            //2.Bottom-up Dynamic Programming + Binary Search
            /*
            Let n be the length of the input string s.
            Time complexity: O(n⋅k⋅logn) Sorting events takes O(nlogn) time. We build dp, a 2D array of size O(n×k) as memory, equal to the number of possible states. Each state is computed with a binary search over all start times, which takes O(logn).
            Space complexity: O(n⋅k) dp takes O(n×k) space.
            */
            maxValue = MaxValueBUDPBS(events, k);

            //3.Top-down Dynamic Programming + Cached Binary Search
            /*
            Let n be the length of the input string s.
            Time complexity:  O(n⋅(k+logn)) Sorting events takes O(nlogn) time. We build a 2D array of size O(n×k) as memory. Each value is computed in O(1) time. The pre-computed table next_indices requires n binary search over the start time in events, 
                              each binary search takes O(logn) time. Therefore the total time it requires is O(n⋅logn).
            Space complexity: O(n⋅k) dp takes O(n×k) space.
            */
            maxValue = MaxValueTDDPCachedBS(events, k);

            //4.Bottom-up Dynamic Programming + Optimized Binary Search
            /*
            Let n be the length of the input string s.
            Time complexity: O(n⋅(k+logn)) Sorting events takes O(nlogn) time. The nested iterations takes n⋅k steps, each step requires O(1) time.Instead of applying binary search in each step, we only have n binary searches, which take n⋅logn time.
            Space complexity: O(n⋅k) dp takes O(n×k) space.
            */
            maxValue = MaxValueBUDPOptiBS(events, k);

            return maxValue;


        }

        private int MaxValueBUDPBS(int[][] events, int k)
        {
            Array.Sort(events, (a, b) => a[0].CompareTo(b[0]));
            int n = events.Length;
            dp = new int[k + 1][];
            for (int curIndex = n - 1; curIndex >= 0; --curIndex)
            {
                for (int count = 1; count <= k; count++)
                {
                    int nextIndex = BisectRight(events, events[curIndex][1]);
                    dp[count][curIndex] = Math.Max(dp[count][curIndex + 1], events[curIndex][2] + dp[count - 1][nextIndex]);
                }
            }
            return dp[k][0];
        }

        private int MaxValueBUDPOptiBS(int[][] events, int k)
        {
            Array.Sort(events, (a, b) => a[0].CompareTo(b[0]));
            int n = events.Length;
            dp = new int[k + 1][];
            for (int curIndex = n - 1; curIndex >= 0; --curIndex)
            {
                int nextIndex = BisectRight(events, events[curIndex][1]);
                for (int count = 1; count <= k; count++)
                {
                    dp[count][curIndex] = Math.Max(dp[count][curIndex + 1], events[curIndex][2] + dp[count - 1][nextIndex]);
                }
            }
            return dp[k][0];
        }

        private int MaxValueTDDPCachedBS(int[][] events, int k)
        {
            Array.Sort(events, (a, b) => a[0].CompareTo(b[0]));
            int n = events.Length;
            dp = new int[k + 1][];
            nextIndices = new int[n];
            for (int curIndex = 0; curIndex < n; ++curIndex)
            {
                nextIndices[curIndex] = BisectRight(events, events[curIndex][1]);
            }
            foreach (int[] row in dp)
            {
                Array.Fill(row, -1);
            }

            return MaxValueTDDPCachedBSDfs(0, k, events);

        }

        private int MaxValueTDDPCachedBSDfs(int curIndex, int count, int[][] events)
        {
            if (count == 0 || curIndex == events.Length)
            {
                return 0;
            }
            if (dp[count][curIndex] != -1)
            {
                return dp[count][curIndex];
            }
            int nextIndex = nextIndices[curIndex];
            dp[count][curIndex] = Math.Max(MaxValueTDDPCachedBSDfs(curIndex + 1, count, events), events[curIndex][2] + MaxValueTDDPCachedBSDfs(nextIndex, count - 1, events));
            return dp[count][curIndex];
        }

        private int[] nextIndices;
        private int MaxValueTDDPBS(int[][] events, int k)
        {
            Array.Sort(events, (a, b) => a[0].CompareTo(b[0]));
            int n = events.Length;
            dp = new int[k + 1][];
            foreach (int[] row in dp)
            {
                Array.Fill(row, -1);
            }
            return MaxValueTDDPBSDfs(0, k, events);

        }

        private int MaxValueTDDPBSDfs(int curIndex, int count, int[][] events)
        {
            if (count == 0 || curIndex == events.Length)
            {
                return 0;
            }
            if (dp[count][curIndex] != -1)
            {
                return dp[count][curIndex];
            }
            int nextIndex = BisectRight(events, events[curIndex][1]);
            dp[count][curIndex] = Math.Max(MaxValueTDDPBSDfs(curIndex + 1, count, events), events[curIndex][2] + MaxValueTDDPBSDfs(nextIndex, count - 1, events));
            return dp[count][curIndex];
        }
        public static int BisectRight(int[][] events, int target)
        {
            int left = 0, right = events.Length;
            while (left < right)
            {
                int mid = (left + right) / 2;
                if (events[mid][0] <= target)
                {
                    left = mid + 1;
                }
                else
                {
                    right = mid;
                }
            }
            return left;
        }

        private int MaxValueNaive(int[][] events, int k)
        {
            Array.Sort(events, (a, b) => a[0].CompareTo(b[0]));
            int n = events.Length;
            dp = new int[k + 1][];
            foreach (int[] row in dp)
            {
                Array.Fill(row, -1);
            }
            return MaxValueNaiveDfs(0, 0, -1, events, k);
        }

        private int MaxValueNaiveDfs(int curIndex, int count, int prevEndingTime, int[][] events, int k)
        {
            if (curIndex == events.Length || count == k)
            {
                return 0;
            }

            if (prevEndingTime >= events[curIndex][0])
            {
                return MaxValueNaiveDfs(curIndex + 1, count, prevEndingTime, events, k);
            }

            if (dp[count][curIndex] != -1)
            {
                return dp[count][curIndex];
            }

            int ans = Math.Max(MaxValueNaiveDfs(curIndex + 1, count, prevEndingTime, events, k),
                               MaxValueNaiveDfs(curIndex + 1, count + 1, events[curIndex][1], events, k) + events[curIndex][2]);
            dp[count][curIndex] = ans;
            return ans;
        }

        int[][] dp; //Memo

        /*
        252. Meeting Rooms
        https://leetcode.com/problems/meeting-rooms/description/
        */
        public bool CanAttendMeetings(int[][] intervals)
        {
            //Brute Force/Naive
            /*
            T: O(n^2) | S: O(1)
            */
            bool canAttendMeetings = CanAttendMeetingsNaive(intervals);

            //Sorting
            /*
            T: O(nlogn) | S: O(1)
            */
            canAttendMeetings = CanAttendMeetingsOptimal(intervals);

            return canAttendMeetings;

        }

        private bool CanAttendMeetingsOptimal(int[][] intervals)
        {
            Array.Sort(intervals, (a, b) => a[0].CompareTo(b[0]));
            for (int i = 0; i < intervals.Length - 1; i++)
            {
                if (intervals[i][1] > intervals[i + 1][0])
                {
                    return false;
                }
            }
            return true;
        }

        private bool CanAttendMeetingsNaive(int[][] intervals)
        {
            for (int i = 0; i < intervals.Length; i++)
            {
                for (int j = i + 1; j < intervals.Length; j++)
                {
                    if (Overlap(intervals[i], intervals[j]))
                    {
                        return false;
                    }
                }
            }
            return true;
        }
        private bool Overlap(int[] interval1, int[] interval2)
        {
            return (interval1[0] >= interval2[0] && interval1[0] < interval2[1])
                || (interval2[0] >= interval1[0] && interval2[0] < interval1[1]);

            //Above condition can be rewritten as follows.
            return (Math.Min(interval1[1], interval2[1]) >
            Math.Max(interval1[0], interval2[0]));
        }

        /*
        253. Meeting Rooms II
        https://leetcode.com/problems/meeting-rooms-ii/
        */
        public int MinMeetingRooms(int[][] intervals)
        {
            //1.Priority Queues
            /*
            Time Complexity: O(NlogN). There are two major portions that take up time here. One is sorting of the array that takes O(NlogN) considering that the array consists of N elements.
                             Then we have the min-heap. In the worst case, all N meetings will collide with each other. In any case we have N add operations on the heap. 
                             In the worst case we will have N extract-min operations as well. Overall complexity being (NlogN) since extract-min operation on a heap takes O(logN).
            Space Complexity: O(N) because we construct the min-heap and that can contain N elements in the worst case as described above in the time complexity section. Hence, the space complexity is O(N).
            */
            int minRooms = MinMeetingRoomsPQ(intervals);

            //2.Chronological Ordering
            /*
            Time Complexity: O(NlogN) because all we are doing is sorting the two arrays for start timings and end timings individually and each of them would contain N elements considering there are N intervals.

            Space Complexity: O(N) because we create two separate arrays of size N, one for recording the start times and one for the end times.

            */

            minRooms = MinMeetingRoomsChronoOrder(intervals);

            return minRooms;

        }

        private int MinMeetingRoomsChronoOrder(int[][] intervals)
        {
            // Check for the base case. If there are no intervals, return 0
            if (intervals.Length == 0)
            {
                return 0;
            }

            int[] start = new int[intervals.Length];
            int[] end = new int[intervals.Length];

            for (int i = 0; i < intervals.Length; i++)
            {
                start[i] = intervals[i][0];
                end[i] = intervals[i][1];
            }
            // Sort the intervals by end time
            Array.Sort(end, (a, b) => a.CompareTo(b));

            // Sort the intervals by start time
            Array.Sort(start, (a, b) => a.CompareTo(b));

            // The two pointers in the algorithm: e_ptr and s_ptr.
            int startPointer = 0, endPointer = 0;

            // Variables to keep track of maximum number of rooms used.
            int usedRooms = 0;

            // Iterate over intervals.
            while (startPointer < intervals.Length)
            {

                // If there is a meeting that has ended by the time the meeting at `start_pointer` starts
                if (start[startPointer] >= end[endPointer])
                {
                    usedRooms -= 1;
                    endPointer += 1;
                }

                // We do this irrespective of whether a room frees up or not.
                // If a room got free, then this used_rooms += 1 wouldn't have any effect. used_rooms would
                // remain the same in that case. If no room was free, then this would increase used_rooms
                usedRooms += 1;
                startPointer += 1;

            }
            return usedRooms;

        }

        private int MinMeetingRoomsPQ(int[][] intervals)
        {
            //sort by start times
            Array.Sort(intervals, (x, y) => x[0] - y[0]); //       var sortedMeetings = intervals.OrderBy(x => x[0]); //Array.Sort
            var pQueue = new PriorityQueue<int, int>();
            var runningMeetings = 0;
            var minRooms = 0;

            foreach (var meeting in intervals)
            {
                while (pQueue.Count > 0 && pQueue.Peek() <= meeting[0])
                {
                    pQueue.Dequeue();
                    runningMeetings--;
                }

                runningMeetings++;
                pQueue.Enqueue(meeting[1], meeting[1]);
                if (minRooms < runningMeetings)
                    minRooms = runningMeetings;
            }
            /*
            Another Variation:

                    var pq = new PriorityQueue<int, int>();

        //  sort by start times
            Array.Sort(intervals, (x,y) => x[0]-y[0]);

            pq.Enqueue(intervals[0][1],intervals[0][1]);

            for(int i=1; i< intervals.Length ; i++){

            //latest meeting ending on or before the current interval
            if(pq.Peek() <= intervals[i][0]){
                pq.Dequeue();
            }

            pq.Enqueue(intervals[i][1],intervals[i][1]);
        }

            */


            return minRooms;

        }

        /*
        2402. Meeting Rooms III
        https://leetcode.com/problems/meeting-rooms-iii/description/
        */
        public int MostBooked(int n, int[][] meetings)
        {
            //1.Sorting and Counting
            /*
            Let N be the number of rooms. Let M be the number of meetings.
            Time complexity: O(M⋅logM+M⋅N). Sorting meetings will incur a time complexity of O(M⋅logM). Iterating over meetings will incur a time complexity of O(M). The inner for loop within the iterations over meetings has a worst-case time complexity of O(N). To illustrate this, envision a scenario where all rooms are initially occupied and remain so throughout the process. In such a case, there is no possibility of breaking out of the loop prematurely.
                            For example: n = 3, meetings = [[1, 10001], [2, 10001], [3, 10001], [4, 10001], [5, 10001], [6, 10001],... [1000, 10001]]
                            In this case, after the first three meetings are assigned to the three rooms, their availability times will be [10001, 10001, 10001]. In this scenario, breaking out of the inner loop early for the remaining meetings becomes unattainable, compelling the algorithm to search for the room that becomes unused earliest. Consequently, the inner loop incurs a worst-case time complexity of O(N). Thus the overall time complexity for iterating over meetings is O(M⋅N). The overall time complexity of the algorithm is O(M⋅logM+M⋅N).
            Space complexity: O(N+sort). Initializing room_availability_time and meeting_count will incur a space complexity of O(N). Some extra space is used when we sort an array of size N in place. The space complexity of the sorting algorithm depends on the programming language.
                            In Python, the sort method sorts a list using the Timsort algorithm which is a combination of Merge Sort and Insertion Sort and has a space complexity of O(N).
                            In C++, the sort() function is implemented as a hybrid of Quick Sort, Heap Sort, and Insertion Sort, with a worst-case space complexity of O(logN).
                            In Java, Arrays.sort() is implemented using a variant of the Quick Sort algorithm which has a space complexity of O(logN).
                            In C#, Array.Sort() method uses the introspective sort (introsort) algorithm as follows:
                                        1.If the partition size is less than or equal to 16 elements, it uses an insertion sort algorithm.
                                        2.If the number of partitions exceeds 2 * LogN, where N is the range of the input array, it uses a Heapsort algorithm.
                                        3.Otherwise, it uses a Quicksort algorithm.

            */
            int maxMeetingCountRoom = MostBookedSortCount(n, meetings);
            //2.Sorting, Counting using Priority Queues
            /*
            Let N be the number of rooms. Let M be the number of meetings.
            Time complexity: O(M⋅logM+M⋅logN). Sorting meetings will incur a time complexity of O(M⋅logM). Popping and pushing into the priority queue will each cost O(logN). These priority queue operations run inside a for loop that runs at most M times leading to a time complexity of O(M⋅logN).
                            The inner nested loop will incur a time complexity of O(logN). The combined time complexity will be O(M⋅logM+M⋅logN). As per the constraints N is small, the term O(M⋅logM) will dominate.
                            Note: Initializing unused_rooms will cost O(N) in ruby and python. But will cost O(N⋅logN) in C++ and Java due to the implementation.
            Space complexity: O(N+sort). Initializing unused_rooms and meeting_count will incur a space complexity of O(N). Some extra space is used when we sort an array of size N in place. The space complexity of the sorting algorithm depends on the programming language.
                            In Python, the sort method sorts a list using the Timsort algorithm which is a combination of Merge Sort and Insertion Sort and has a space complexity of O(N).
                            In C++, the sort() function is implemented as a hybrid of Quick Sort, Heap Sort, and Insertion Sort, with a worst-case space complexity of O(logN).
                            In Java, Arrays.sort() is implemented using a variant of the Quick Sort algorithm which has a space complexity of O(logN).
                            In C#, Array.Sort() method uses the introspective sort (introsort) algorithm as follows:
                                        1.If the partition size is less than or equal to 16 elements, it uses an insertion sort algorithm.
                                        2.If the number of partitions exceeds 2 * LogN, where N is the range of the input array, it uses a Heapsort algorithm.
                                        3.Otherwise, it uses a Quicksort algorithm.



            */
            maxMeetingCountRoom = MostBookedSortCountPQ(n, meetings);

            return maxMeetingCountRoom;

        }

        private int MostBookedSortCountPQ(int n, int[][] meetings)
        {
            // Counter for number of meetings held in room
            int[] rooms = new int[n];

            // List(start, end, room number)
            PriorityQueue<List<long>, long> occupiedRooms = new PriorityQueue<List<long>, long>(Comparer<long>.Create((x, y) => x.CompareTo(y)));

            // 1. Each meeting will take place in the free room with the lowest number.
            PriorityQueue<long, long> freeRooms = new PriorityQueue<long, long>();
            for (int i = 0; i < n; i++)
            {
                freeRooms.Enqueue(i, i);
            }

            // 3. Meetings that have an earlier original start time should be given the room
            Array.Sort(meetings, (x, y) => x[0].CompareTo(y[0]));
            long currentTime = 0;
            for (int i = 0; i < meetings.Length; i++)
            {
                int[] meeting = meetings[i];
                // Update time to meeting time if meeting time is later
                currentTime = Math.Max(meeting[0], currentTime);

                // If no meeting rooms left, go to time where earliest room will be cleared
                if (freeRooms.Count == 0)
                {
                    long earliestFreeTime = occupiedRooms.Peek()[1];
                    currentTime = Math.Max(earliestFreeTime, currentTime);
                }

                // Clear all rooms occuring at and before this time
                while (occupiedRooms.Count > 0 && occupiedRooms.Peek()[1] <= currentTime)
                {
                    long freedRoom = occupiedRooms.Dequeue()[2];
                    freeRooms.Enqueue(freedRoom, freedRoom);
                }

                // Occupy a new room, 
                // 2. Delayed meeting should have same duration
                long nextRoom = freeRooms.Dequeue();
                rooms[nextRoom] += 1;
                occupiedRooms.Enqueue(new List<long> { currentTime, currentTime + (meeting[1] - meeting[0]), nextRoom }, currentTime + (meeting[1] - meeting[0]));
            }

            // Get smallest room with largest meetings held
            int maxMeetingCount = 0, maxMeetingCountRoom = 0;
            for (int i = n - 1; i >= 0; i--)
            {
                if (rooms[i] >= maxMeetingCount)
                {
                    maxMeetingCount = rooms[i];
                    maxMeetingCountRoom = i;
                }
            }
            return maxMeetingCountRoom;
        }

        private int MostBookedSortCount(int n, int[][] meetings)
        {
            long[] roomAvailabilityTime = new long[n];
            int[] meetingCount = new int[n];
            Array.Sort(meetings, (a, b) => a[0].CompareTo(b[0]));

            foreach (int[] meeting in meetings)
            {
                int start = meeting[0], end = meeting[1];
                long minRoomAvailabilityTime = long.MaxValue;
                int minAvailableTimeRoom = 0;
                bool foundUnusedRoom = false;

                for (int i = 0; i < n; i++)
                {
                    if (roomAvailabilityTime[i] <= start)
                    {
                        foundUnusedRoom = true;
                        meetingCount[i]++;
                        roomAvailabilityTime[i] = end;
                        break;
                    }

                    if (minRoomAvailabilityTime > roomAvailabilityTime[i])
                    {
                        minRoomAvailabilityTime = roomAvailabilityTime[i];
                        minAvailableTimeRoom = i;
                    }
                }

                if (!foundUnusedRoom)
                {
                    roomAvailabilityTime[minAvailableTimeRoom] += end - start;
                    meetingCount[minAvailableTimeRoom]++;
                }
            }

            int maxMeetingCount = 0, maxMeetingCountRoom = 0;
            for (int i = 0; i < n; i++)
            {
                if (meetingCount[i] > maxMeetingCount)
                {
                    maxMeetingCount = meetingCount[i];
                    maxMeetingCountRoom = i;
                }
            }

            return maxMeetingCountRoom;
        }
        /*
        56. Merge Intervals
        https://leetcode.com/problems/merge-intervals/description/
        */
        public int[][] MergeIntervals(int[][] intervals)
        {
            //1.Connected Components
            /*
            Time complexity : O(n^2)  Building the graph costs O(V+E)=O(V)+O(E)=O(n)+O(n^2)=O(n^2) time, as in the worst case all intervals are mutually overlapping. 
                            Traversing the graph has the same cost (although it might appear higher at first) because our visited set guarantees that each node will be visited exactly once.
                            Finally, because each node is part of exactly one component, the merge step costs O(V)=O(n) time. This all adds up as follows:
                            O(n^2)+O(n^2)+O(n) = O(n^2)

            Space complexity : O(n^2) As previously mentioned, in the worst case, all intervals are mutually overlapping, so there will be an edge for every pair of intervals. Therefore, the memory footprint is quadratic in the input size.

            */
            List<int[]> merged = new List<int[]>();
            merged = MergeIntervalsNaive(intervals);

            //2.Sorting
            /*
            Time complexity : O(nlogn) Other than the sort invocation, we do a simple linear scan of the list, so the runtime is dominated by the O(nlogn) complexity of sorting.
            Space complexity : O(logN) (or O(n)) If we can sort intervals in place, we do not need more than constant additional space, although the sorting itself takes O(logn) space. Otherwise, we must allocate linear space to store a copy of intervals and sort that.
            */
            merged = MergeIntervalsOptimal(intervals);

            return merged.ToArray();

        }

        private List<int[]> MergeIntervalsOptimal(int[][] intervals)
        {
            Array.Sort(intervals, (a, b) => a[0] - b[0]);
            LinkedList<int[]> merged = new LinkedList<int[]>();
            foreach (int[] interval in intervals)
            {
                // if the list of merged intervals is empty or if the current
                // interval does not overlap with the previous, append it
                if (merged.Count == 0 || merged.Last.Value[1] < interval[0])
                {
                    merged.AddLast(interval);
                }
                // otherwise, there is overlap, so we merge the current and previous
                // intervals
                else
                {
                    merged.Last.Value[1] =
                        Math.Max(merged.Last.Value[1], interval[1]);
                }
            }

            return merged.ToList();
        }
        //https://www.algoexpert.io/questions/merge-overlapping-intervals
        //Movie watch intervals
        // O(nlog(n)) time | O(n) space - where n is the length of the input array
        public int[][] MergeOverlappingIntervals(int[][] intervals)
        {
            // Sort the intervals by starting value.
            int[][] sortedIntervals = intervals.Clone() as int[][];
            Array.Sort(sortedIntervals, (a, b) => a[0].CompareTo(b[0]));

            List<int[]> mergedIntervals = new List<int[]>();
            int[] currentInterval = sortedIntervals[0];
            mergedIntervals.Add(currentInterval);

            foreach (var nextInterval in sortedIntervals)
            {
                int currentIntervalEnd = currentInterval[1];
                int nextIntervalStart = nextInterval[0];
                int nextIntervalEnd = nextInterval[1];

                if (currentIntervalEnd >= nextIntervalStart)
                {
                    currentInterval[1] = Math.Max(currentIntervalEnd, nextIntervalEnd);
                }
                else
                {
                    currentInterval = nextInterval;
                    mergedIntervals.Add(currentInterval);
                }
            }

            return mergedIntervals.ToArray();
        }


        private bool Overlap1(int[] a, int[] b)
        {
            return a[0] <= b[1] && b[0] <= a[1];
        }
        private void BuildGraph(int[][] intervals)
        {
            foreach (var interval in intervals)
            {
                graph[interval] = new List<int[]>();
            }

            for (int i = 0; i < intervals.Length; i++)
            {
                for (int j = 0; j < intervals.Length; j++)
                {
                    if (this.Overlap1(intervals[i], intervals[j]))
                    {
                        graph.TryGetValue(intervals[i], out var val);
                        if (val != null)
                        {
                            val.Add(intervals[j]);
                        }

                        graph.TryGetValue(intervals[j], out val);
                        if (val != null)
                        {
                            val.Add(intervals[i]);
                        }
                    }
                }
            }
        }
        IDictionary<int[], List<int[]>> graph =
                new Dictionary<int[], List<int[]>>();

        IDictionary<int, List<int[]>> nodesInComp =
            new Dictionary<int, List<int[]>>();

        HashSet<int[]> visitedSet = new HashSet<int[]>();
        private List<int[]> MergeIntervalsNaive(int[][] intervals)
        {

            BuildGraph(intervals);
            BuildComponents(intervals);
            List<int[]> merged = new List<int[]>();
            for (int i = 0; i < nodesInComp.Count; i++)
            {
                merged.Add(MergeNodes(nodesInComp[i]));
            }

            return merged;
        }

        private int[] MergeNodes(List<int[]> nodes)
        {
            int minStart = nodes[0][0];
            foreach (var node in nodes)
            {
                minStart = Math.Min(minStart, node[0]);
            }

            int maxEnd = nodes[0][1];
            foreach (var node in nodes)
            {
                maxEnd = Math.Max(maxEnd, node[1]);
            }

            return new int[] { minStart, maxEnd };
        }

        private void MarkComponentDFS(int[] start, Stack<int[]> stack,
                                      int compNumber)
        {
            stack.Push(start);
            while (stack.Count != 0)
            {
                int[] node = stack.Pop();
                if (!visitedSet.Contains(node))
                {
                    visitedSet.Add(node);
                    if (!nodesInComp.ContainsKey(compNumber))
                    {
                        nodesInComp.Add(compNumber, new List<int[]>());
                    }

                    nodesInComp.TryGetValue(compNumber, out var val);
                    val.Add(node);
                    List<int[]> nodes = null;
                    graph.TryGetValue(node, out nodes);
                    foreach (var child in nodes)
                    {
                        stack.Push(child);
                    }
                }
            }
        }

        private void BuildComponents(int[][] intervals)
        {
            int compNumber = 0;
            Stack<int[]> stack = new Stack<int[]>();
            foreach (var interval in intervals)
            {
                if (!visitedSet.Contains(interval))
                {
                    MarkComponentDFS(interval, stack, compNumber);
                    compNumber++;
                }
            }
        }
        /*
        759. Employee Free Time
        https://leetcode.com/problems/employee-free-time/description/
        
        Using Priority Queue and without PQ
        Time complexity: O(n∗logk)
        Space complexity: O(n)
        */
        public List<Interval> EmployeeFreeTimePQ(IList<IList<Interval>> schedule)
        {
            var pq = new PriorityQueue<(int person, int index), int>();
            for (int i = 0; i < schedule.Count; i++)
            {
                pq.Enqueue((i, 0), schedule[i][0].Start);
            }
            List<Interval> res = new();
            int prev = schedule[pq.Peek().person][pq.Peek().index].Start;
            while (pq.Count > 0)
            {
                (int person, int index) = pq.Dequeue();
                Interval interval = schedule[person][index];
                if (interval.Start > prev)
                {
                    res.Add(new Interval(prev, interval.Start));
                }
                prev = Math.Max(prev, interval.End);
                if (schedule[person].Count > index + 1)
                {
                    pq.Enqueue((person, index + 1), schedule[person][index + 1].Start);
                }
            }
            return res;
        }
        public List<Interval> EmployeeFreeTime(List<List<Interval>> avails)
        {
            List<Interval> freeTimeIntervals = new List<Interval>();
            List<Interval> allIntervals = new List<Interval>();
            avails.ForEach(e => allIntervals.AddRange(e));
            allIntervals.Sort((a, b) => a.Start.CompareTo(b.Start));

            Interval temp = allIntervals[0];
            foreach (Interval each in allIntervals)
            {
                if (temp.End < each.Start)
                {
                    freeTimeIntervals.Add(new Interval(temp.End, each.Start));
                    temp = each;
                }
                else
                {
                    temp = temp.End < each.End ? each : temp;
                }
            }
            return freeTimeIntervals;
        }

        public class Interval
        {
            public int Start;
            public int End;

            public Interval() { }
            public Interval(int _start, int _end)
            {
                Start = _start;
                End = _end;
            }
        }
        /*
        3169. Count Days Without Meetings
        https://leetcode.com/problems/count-days-without-meetings/description/
        
        •	Time complexity: O(nlogn)
        •	Space complexity: O(1)

        */
        public int CountDays(int days, int[][] meetings)
        {
            Array.Sort(meetings, (a, b) => a[0].CompareTo(b[0]));
            int res = 0;
            int maxEnd = meetings[0][1];
            for (int i = 1; i < meetings.Length; i++)
            {

                if (meetings[i][0] <= maxEnd)
                {
                    maxEnd = Math.Max(maxEnd, meetings[i][1]);
                }

                if (meetings[i][0] > maxEnd)
                {
                    res += (meetings[i][0] - maxEnd - 1);
                    maxEnd = meetings[i][1];
                }

            }
            if (meetings[0][0] > 1)
            {
                res += meetings[0][0] - 1;
            }
            Array.Sort(meetings, (a, b) => a[1].CompareTo(b[1]));
            if (meetings[meetings.Length - 1][1] < days)
            {
                res += days - meetings[meetings.Length - 1][1];
            }
            return res;
        }

        /*
        2446. Determine if Two Events Have Conflict
        https://leetcode.com/problems/determine-if-two-events-have-conflict/description/

        Time O(1)
        Space O(1)

        */
        public bool HaveConflict(string[] event1, string[] event2)
            => event1[0].CompareTo(event2[1]) <= 0 &&
            event2[0].CompareTo(event1[1]) <= 0;



        /*
        729. My Calendar I
        https://leetcode.com/problems/my-calendar-i/
        */

        /*Approach #1: Brute Force
        Let N be the number of events booked.
        Time Complexity: O(N^2). For each new event, we process every previous event to decide whether the new event can be booked. This leads to ∑^Nk= O(k)=O(N^2) complexity.
        Space Complexity: O(N), the size of the calendar.        */
        public class MyCalendarNaive
        {
            List<int[]> calendar;


            public MyCalendarNaive()
            {
                calendar = new List<int[]>();
            }

            public bool Book(int start, int end)
            {
                foreach (int[] iv in calendar)
                {
                    if (iv[0] < end && start < iv[1])
                    {
                        return false;
                    }
                }
                calendar.Add(new int[] { start, end });
                return true;

            }
        }
        /*Approach #2: Brute Force
        Let N be the number of events booked.
        Time Complexity: O(NlogN). For each new event, we search that the event is legal in O(logN) time, then insert it in O(logN) time.
        Space Complexity: O(N), the size of the data structures used.

            /**
         * Your MyCalendar object will be instantiated and called as such:
         * MyCalendar obj = new MyCalendar();
         * bool param_1 = obj.Book(start,end);
         
        */



        class MyCalendarOptimal
        {
            //TreeMap<Integer, Integer> calendar;
            private SortedDictionary<int, int> calendar;

            MyCalendarOptimal()
            {
                calendar = new SortedDictionary<int, int>();

            }

            public bool Book(int start, int end)
            {
                int? previousKey = null;
                int? nextKey = null;

                foreach (var key in calendar.Keys)
                {
                    if (key <= start)
                    {
                        previousKey = key;
                    }
                    else
                    {
                        nextKey = key;
                        break;
                    }
                }

                if ((previousKey == null || calendar[previousKey.Value] <= start) &&
                    (nextKey == null || end <= nextKey))
                {
                    calendar[start] = end;
                    return true;
                }
                return false;
            }
        }

        /*
        731. My Calendar II
       https://leetcode.com/problems/my-calendar-ii/description/

        */
        public class MyCalendarTwo
        {
            private SortedDictionary<int, int> map;

            public MyCalendarTwo()
            {
                map = new SortedDictionary<int, int>();
            }

            public bool Book(int start, int end)
            {
                map[start] = map.GetValueOrDefault(start, 0) + 1;
                map[end] = map.GetValueOrDefault(end, 0) - 1;
                int count = 0;

                foreach (var entry in map)
                {
                    count += entry.Value;
                    if (count > 2)
                    {
                        map[start] = map[start] - 1;
                        if (map[start] == 0)
                        {
                            map.Remove(start);
                        }
                        map[end] = map.GetValueOrDefault(end, 0) + 1;
                        if (map[end] == 0)
                        {
                            map.Remove(end);
                        }
                        return false;
                    }
                }
                return true;
            }
        }

 
       /*
        2462. Total Cost to Hire K Workers
        https://leetcode.com/problems/total-cost-to-hire-k-workers/description

        */
        public long TotalCost(int[] costs, int k, int candidates)
        {
            //1. 2 Priority Queues
            /*
            Let m be the given integer candidates.
            Time complexity: O((k+m)⋅logm) We need to initialize two priority queues of size m, which takes O(m⋅logm) time.
                             During the hiring rounds, we keep removing the top element from priority queues and adding new elements for up to k times. Operations on a priority queue take amortized O(logm) time. 
                             Thus this process takes O(k⋅logm) time. Note: in Python, heapq.heapify() creates the priority queue in linear time. Therefore, in Python, the time complexity is O(m+k⋅logm).
            Space complexity: O(m) We need to store the first m and the last m workers in two priority queues.
            */
            long totalCost = TotalCost2PQ(costs, k, candidates);

            //2. Single Priority Queue
            /*
            For the sake of brevity, let m be the given integer candidates.
            Time complexity: O((k+m)⋅logm) We need to initialize one priority queue pq of size up to 2⋅m, which takes O(m⋅logm) time.
                            During k hiring rounds, we keep popping top elements from pq and pushing new elements into pq for up to k times. Operations on a priority queue take amortized O(logm) time. Thus this process takes O(k⋅logm) time.
                            Note: in Python, heapq.heapify() creates the priority queue in linear time. Therefore, in Python, the time complexity is O(m+k⋅logm).
            Space complexity: O(m)  We need to store at most 2⋅m elements (the first m and the last m elements) of costs in the priority queue pq.
            */
            totalCost = TotalCost1PQ(costs, k, candidates);

            return totalCost;

        }

        private long TotalCost2PQ(int[] costs, int k, int candidates)
        {
            PriorityQueue<int, int> headWorkers = new PriorityQueue<int, int>();
            PriorityQueue<int, int> tailWorkers = new PriorityQueue<int, int>();
            // headWorkers stores the first k workers.
            // tailWorkers stores at most last k workers without any workers from the first k workers.
            for (int i = 0; i < candidates; i++)
            {
                headWorkers.Enqueue(costs[i], costs[i]);
            }
            for (int i = Math.Max(candidates, costs.Length - candidates); i < costs.Length; i++)
            {
                tailWorkers.Enqueue(costs[i], costs[i]);
            }

            long totalCost = 0;
            int nextHead = candidates;
            int nextTail = costs.Length - 1 - candidates;

            for (int i = 0; i < k; i++)
            {
                if (tailWorkers.Count == 0 || (headWorkers.Count > 0 && headWorkers.Peek() <= tailWorkers.Peek()))
                {
                    totalCost += headWorkers.Dequeue();

                    // Only refill the queue if there are workers outside the two queues.
                    if (nextHead <= nextTail)
                    {
                        headWorkers.Enqueue(costs[nextHead], costs[nextHead]);
                        nextHead++;
                    }
                }
                else
                {
                    totalCost += tailWorkers.Dequeue();

                    // Only refill the queue if there are workers outside the two queues.
                    if (nextHead <= nextTail)
                    {
                        tailWorkers.Enqueue(costs[nextTail], costs[nextTail]);
                        nextTail--;
                    }
                }
            }

            return totalCost;

        }

        private long TotalCost1PQ(int[] costs, int k, int candidates)
        {
            var left = 0;
            var right = costs.Length - 1;
            var candidatePq = new PriorityQueue<Tuple<int, int>, Tuple<int, int>>(new TupleComparer());

            while (left < candidates)
            {
                var tuple = new Tuple<int, int>(costs[left], left);
                candidatePq.Enqueue(tuple, tuple);
                left++;
            }
            while (right >= left && right >= costs.Length - candidates)
            {
                var tuple = new Tuple<int, int>(costs[right], right);
                candidatePq.Enqueue(tuple, tuple);
                right--;
            }

            var selectedcandidate = 0;
            long cost = 0;
            while (selectedcandidate < k)
            {
                var candidate = candidatePq.Dequeue();
                cost += candidate.Item1;

                var index = candidate.Item2;
                if (index < left && right >= left)
                {
                    var tuple = new Tuple<int, int>(costs[left], left);
                    candidatePq.Enqueue(tuple, tuple);
                    left++;
                }
                else if (index > right && right >= left)
                {
                    var tuple = new Tuple<int, int>(costs[right], right);
                    candidatePq.Enqueue(tuple, tuple);
                    right--;
                }
                selectedcandidate++;
            }
            return cost;

        }
        private class TupleComparer : IComparer<Tuple<int, int>>
        {
            // Compares by Height, Length, and Width.
            public int Compare(Tuple<int, int> x, Tuple<int, int> y)
            {
                if (x.Item1.CompareTo(y.Item1) != 0)
                {
                    return x.Item1.CompareTo(y.Item1);
                }
                else if (x.Item2.CompareTo(y.Item2) != 0)
                {
                    return x.Item2.CompareTo(y.Item2);
                }
                else
                {
                    return 0;
                }
            }
        }
        /*
        1606. Find Servers That Handled Most Number of Requests
        https://leetcode.com/problems/find-servers-that-handled-most-number-of-requests/description/

        */
        public IList<int> BusiestServers(int numberOfServers, int[] arrival, int[] load)
        {
            //1.Sorted Containers
            /*
            Let k be the number of servers and n be the size of the input array arrival, that is, the number of requests.

            Time complexity:O(n⋅logk) We used a priority queue busy and a sorted container free.
                            Operations like adding and removing in the priority queue take logarithmic time. Since there may be at most k servers stored in the priority queue, thus each operation takes O(logk) time.
                            Sorted containers are implemented using a red-black tree, and operations like inserting, deleting, and performing a binary search on the red-black tree take O(logk) time.
                            In each step, we perform multiple operations on busy and free. Therefore, the overall time complexity is O(n⋅logk).
            Space complexity: O(k) The total number of servers stored in busy and free is n, so they take O(k) space.
                            We used an array count to record the number of requests handled by each server, which takes O(k) space.
                            To sum up, the overall time complexity is O(k).

            */
            IList<int> result = BusiestServersWithSortedContainers(numberOfServers, arrival, load);

            //2. Two Priority Queues
            /*
            Let k be the number of servers and n be the size of the input array arrival, that is, the number of requests.

            Time complexity:O(n⋅logk) We used two priority queues named busy and free to store all servers, each operation like adding and removing in a priority queue of size O(k) takes O(logk) time.
                            In each iteration step, we make several operations on busy and free that take O(logk) time.
                            Therefore, the overall time complexity is O(n⋅logk).
            Space complexity: O(k) We used two priority queues named busy and free to store all servers, that take O(k) space.
                              We used an array count to record the number of requests handled by each server, which also takes O(k) space.
                              To sum up, the overall time complexity is O(k).
            */

            result = BusiestServers2PQ(numberOfServers, arrival, load);

            return result;

        }

        private IList<int> BusiestServers2PQ(int numberOfServers, int[] arrival, int[] load)
        {
            throw new NotImplementedException();
            //TODO: Convert Below Java to C#
            /*
            int[] count = new int[k];
                PriorityQueue<Integer> free = new PriorityQueue<>((a, b) -> a - b);
                PriorityQueue<Pair<Integer, Integer>> busy = new PriorityQueue<>((a, b) -> a.getKey() - b.getKey());
                
                // All servers are free at the beginning.

                for (int i = 0; i < k; ++i) {
                    free.add(i);
                }
                
                for (int i = 0; i < arrival.length; ++i) {
                    int start = arrival[i];

                    // Remove free servers from 'busy', modify their IDs and
                    // add them to 'free'
                    while (!busy.isEmpty() && busy.peek().getKey() <= start) {
                        Pair<Integer, Integer> curr = busy.remove();
                        int serverId = curr.getValue();
                        int modifiedId = ((serverId - i) % k + k) % k + i;
                        free.add(modifiedId);
                    }

                    // Get the original server ID by taking the remainder of
                    // the modified ID to k.
                    if (!free.isEmpty()) {
                        int busyId = free.peek() % k;
                        free.remove();
                        busy.add(new Pair<>(start + load[i], busyId));
                        count[busyId]++;
                    }
                }
                
                // Find the servers that have the maximum workload.
                int maxJob = Collections.max(Arrays.stream(count).boxed().collect(Collectors.toList()));
                List<Integer> answer = new ArrayList<>();
                for (int i = 0; i < k; ++i) {
                    if (count[i] == maxJob) {
                        answer.add(i);
                    }
                }
                
                return answer;
            */
        }

        private IList<int> BusiestServersWithSortedContainers(int numberOfServers, int[] arrival, int[] load)
        {
            int[] serverCount = new int[numberOfServers];
            SortedSet<int> freeServers = new SortedSet<int>();
            PriorityQueue<(int, int), int> busyServers = new PriorityQueue<(int, int), int>();

            // All servers are free at the beginning.
            for (int i = 0; i < numberOfServers; ++i)
            {
                freeServers.Add(i);
            }

            for (int i = 0; i < arrival.Length; ++i)
            {
                int startTime = arrival[i];

                // Move free servers from 'busy' to 'free'.
                while (busyServers.Count > 0 && busyServers.Peek().Item1 <= startTime)
                {
                    var current = busyServers.Dequeue();
                    int serverId = current.Item2;
                    freeServers.Add(serverId);
                }

                // If we have free servers, use binary search to find the 
                // target server.
                if (freeServers.Count > 0)
                {
                    int busyServerId = freeServers.FirstOrDefault(s => s >= i % numberOfServers);
                    if (busyServerId == default)
                    {
                        busyServerId = freeServers.Min;
                    }

                    freeServers.Remove(busyServerId);
                    busyServers.Enqueue((startTime + load[i], busyServerId), busyServerId);
                    serverCount[busyServerId]++;
                }
            }

            // Find the servers that have the maximum workload.
            int maxJobCount = serverCount.Max();
            List<int> result = new List<int>();
            for (int i = 0; i < numberOfServers; ++i)
            {
                if (serverCount[i] == maxJobCount)
                {
                    result.Add(i);
                }
            }

            return result;

        }
        /*
        1094. Car Pooling
        https://leetcode.com/problems/car-pooling/description
        */
        public bool CarPooling(int[][] trips, int capacity)
        {
            //1. Time Stamp
            /*
            Assume N is the length of trips.
            Time complexity: O(Nlog(N)) since we need to iterate over trips and sort our timestamp. Iterating costs O(N), and sorting costs O(Nlog(N)), and adding together we have O(N)+O(Nlog(N))=O(Nlog(N)).
            Space complexity: O(N) since in the worst case we need O(N) to store timestamp.
            */
            bool canPickDropAll = CarPooling1(trips, capacity);

            //2. Bucket Sort
            /*
            Assume N is the length of trip.
            Time complexity: O(max(N,1001)) since we need to iterate over trips and then iterate over our 1001 buckets.
            Space complexity : O(1001)=O(1) since we have 1001 buckets.


            */
            canPickDropAll = CarPoolingWithBucketSort(trips, capacity);

            return canPickDropAll;

        }

        private bool CarPoolingWithBucketSort(int[][] trips, int capacity)
        {
            int[] timestamp = new int[1001];
            foreach (int[] trip in trips)
            {
                timestamp[trip[1]] += trip[0];
                timestamp[trip[2]] -= trip[0];
            }
            int usedCapacity = 0;
            foreach (int number in timestamp)
            {
                usedCapacity += number;
                if (usedCapacity > capacity)
                {
                    return false;
                }
            }
            return true;
        }


        public bool CarPooling1(int[][] trips, int capacity)
        {
            SortedDictionary<int, int> passengerChanges = new SortedDictionary<int, int>();
            foreach (int[] trip in trips)
            {
                int startPassenger = passengerChanges.GetValueOrDefault(trip[1], 0) + trip[0];
                passengerChanges[trip[1]] = startPassenger;

                int endPassenger = passengerChanges.GetValueOrDefault(trip[2], 0) - trip[0];
                passengerChanges[trip[2]] = endPassenger;
            }
            int usedCapacity = 0;
            foreach (int passengerChange in passengerChanges.Values)
            {
                usedCapacity += passengerChange;
                if (usedCapacity > capacity)
                {
                    return false;
                }
            }
            return true;
        }
        /*
        2251. Number of Flowers in Full Bloom
        https://leetcode.com/problems/number-of-flowers-in-full-bloom/description/
        */

        public int[] FullBloomFlowers(int[][] flowers, int[] people)
        {
            //Approach 1: Heap/Priority Queue
            /*
            Given n as the length of flowers and m as the length of people,
            Time complexity: O(n⋅logn+m⋅(logn+logm))  We start by sorting both flowers and people. This costs O(n⋅logn) and O(m⋅logm) respectively. Next, we perform O(m) iterations. 
                            At each iteration, we perform some heap operations. The cost of these operations is dependent on the size of the heap. Our heap cannot exceed a size of n, so these operations cost O(logn).
                            There are some other linear time operations that don't affect our time complexity. In total, our time complexity is O(n⋅logn+m⋅(logn+logm)).
            Space complexity: O(n+m) We create an array sortedPeople of length m. dic also grows to a length of m, and heap can grow to a size of O(n).

            */
            int[] result = FullBloomFlowersPQ(flowers, people);

            //Approach 2: Difference Array + Binary Search
            /*
            Given n as the length of flowers and m as the length of people,

            Time complexity: O((n+m)⋅logn) Our first loop sets difference, which costs O(n⋅logn).Next, we calculate the prefix sum, which will cost either O(n) or O(n⋅logn) depending on your language's implementation. 
                            This is because difference will have a size between n and 2n. Finally, we have a loop over people. We perform a binary search that costs O(logn) at each iteration. Thus, we spend m⋅logn here.
                            This gives us a final time complexity of O((n+m)⋅logn)
            Space complexity: O(n) difference has a size of O(n). prefix and positions have the same size as difference.
            */
            result = FullBloomFlowersBS(flowers, people);

            //Approach 3: Simpler Binary Search
            /*
            Given n as the length of flowers and m as the length of people,

            Time complexity: O((n+m)⋅logn) We first create two arrays of length n, starts and ends, then sort them. This costs O(n⋅logn).
                            Next, we iterate over people and perform two binary searches at each iteration. This costs O(m⋅logn).
                            Thus, our time complexity is O((n+m)⋅logn).

            Space complexity: O(n) starts and ends both have a size of n.
            */
            result = FullBloomFlowersOptimalBS(flowers, people);

            return result;


        }

        private int[] FullBloomFlowersOptimalBS(int[][] flowers, int[] people)
        {
            List<int> starts = new List<int>();
            List<int> ends = new List<int>();

            foreach (int[] flower in flowers)
            {
                starts.Add(flower[0]);
                ends.Add(flower[1] + 1);
            }

            starts.Sort();
            ends.Sort();
            int[] ans = new int[people.Length];

            for (int index = 0; index < people.Length; index++)
            {
                int person = people[index];
                int i = BinarySearch(starts, person);
                int j = BinarySearch(ends, person);
                ans[index] = i - j;
            }

            return ans;
        }

        private int[] FullBloomFlowersBS(int[][] flowers, int[] people)
        {
            SortedDictionary<int, int> difference = new SortedDictionary<int, int>();
            difference[0] = 0;

            foreach (int[] flower in flowers)
            {
                int start = flower[0];
                int end = flower[1] + 1;

                difference[start] = difference.GetValueOrDefault(start, 0) + 1;
                difference[end] = difference.GetValueOrDefault(end, 0) - 1;
            }

            List<int> positions = new List<int>();
            List<int> prefix = new List<int>();
            int currentCount = 0;

            foreach (int key in difference.Keys)
            {
                positions.Add(key);
                currentCount += difference[key];
                prefix.Add(currentCount);
            }

            int[] answer = new int[people.Length];
            for (int j = 0; j < people.Length; j++)
            {
                int i = BinarySearch(positions, people[j]) - 1;
                answer[j] = prefix[i];
            }

            return answer;
        }
        private int BinarySearch(List<int> arr, int target)
        {
            int left = 0;
            int right = arr.Count;
            while (left < right)
            {
                int mid = (left + right) / 2;
                if (target < arr[mid])
                {
                    right = mid;
                }
                else
                {
                    left = mid + 1;
                }
            }

            return left;
        }
        private int[] FullBloomFlowersPQ(int[][] flowers, int[] people)
        {
            int[] sortedPeople = new int[people.Length];
            //Array.Copy(people, sortedPeople, people.Length);            
            sortedPeople = (int[])people.Clone();
            Array.Sort(sortedPeople);

            Array.Sort(flowers, (a, b) => a[0].CompareTo(b[0]));
            //Array.Sort(flowers, (a, b) => Enumerable.SequenceEqual(a, b) ? 0 : (a[0] < b[0] ? -1 : 1));

            Dictionary<int, int> dict = new Dictionary<int, int>();
            PriorityQueue<int, int> heap = new PriorityQueue<int, int>();

            int idx = 0;
            foreach (int person in sortedPeople)
            {
                while (idx < flowers.Length && flowers[idx][0] <= person)
                {
                    heap.Enqueue(flowers[idx][1], flowers[idx][1]);
                    idx++;
                }

                while (heap.Count > 0 && heap.Peek() < person)
                {
                    heap.Dequeue();
                }

                dict.Add(person, heap.Count);
            }

            int[] ans = new int[people.Length];
            for (int j = 0; j < people.Length; j++)
            {
                ans[j] = dict[people[j]];
            }

            return ans;
        }
        /*
        452. Minimum Number of Arrows to Burst Balloons
        https://leetcode.com/problems/minimum-number-of-arrows-to-burst-balloons/description/
        •	Time complexity : O(NlogN) because of sorting of the input data.
        •	Space complexity : O(N) or O(logN)
                            o	The space complexity of the sorting algorithm depends on the implementation of each programming language.
                            o	For instance, the list.sort() function in Python is implemented with the Timsort algorithm whose space complexity is O(N).
                            o	In Java, the Arrays.sort() is implemented as a variant of quicksort algorithm whose space complexity is O(logN).
                            o	In C#, the Array.Sort() is implemented as an Introsprective sort(introsort) algo which a combination of insertion sort, heap sort and quicksort algorithm based on partitions whose space complexity is O(logN).
        */
        public int FindMinArrowShots(int[][] points)
        {
            if (points.Length == 0) return 0;

            // sort by x_end
            Array.Sort(points, (o1, o2) =>
            {
                // We can't simply use the o1[1] - o2[1] trick, as this will cause an 
                // integer overflow for very large or small values.
                if (o1[1] == o2[1]) return 0;
                if (o1[1] < o2[1]) return -1;
                return 1;
            });

            int arrows = 1;
            int xStart, xEnd, firstEnd = points[0][1];
            foreach (int[] p in points)
            {
                xStart = p[0];
                xEnd = p[1];

                // If the current balloon starts after the end of another one,
                // one needs one more arrow
                if (firstEnd < xStart)
                {
                    arrows++;
                    firstEnd = xEnd;
                }
            }

            return arrows;
        }
        /*
        435. Non-overlapping Intervals
        https://leetcode.com/problems/non-overlapping-intervals
        Time complexity: O(n⋅logn) : We sort intervals, which costs O(n⋅logn). Then, we iterate over the input, performing constant time work at each iteration. This means the iteration costs O(n), which is dominated by the sort.
        Space Complexity: O(logn) or O(n):       The space complexity of the sorting algorithm depends on the implementation of each programming language:
        */
        public int EraseOverlapIntervals(int[][] intervals)
        {
            Array.Sort(intervals, (a, b) => a[1].CompareTo(b[1]));
            int ans = 0;
            int k = int.MinValue;

            for (int i = 0; i < intervals.Length; i++)
            {
                int x = intervals[i][0];
                int y = intervals[i][1];

                if (x >= k)
                {
                    // Case 1
                    k = y;
                }
                else
                {
                    // Case 2
                    ans++;
                }
            }

            return ans;

        }
        /*
        2658. Maximum Number of Fish in a Grid
        https://leetcode.com/problems/maximum-number-of-fish-in-a-grid/description/

        */
        public int FindMaxFish(int[][] grid)
        {
            //1. DFS
            /*
            Time: O(N * M) you are traversing entire matrix
            Space: O(N * M) that is size of a call stack
            */
            int maxFish = FindMaxFishDfs(grid);

            //2. BFS
            /*
            Time: O(N * M) you are traversing entire matrix
            Space: O(N * M)  that is the size of a queue, 
            */
            maxFish = FindMaxFishBfs(grid);

            return maxFish;

        }

        private int FindMaxFishBfs(int[][] grid)
        {
            int n = grid.Length;
            int m = grid[0].Length;
            int maxFish = 0; // variable to store the maximum number of fish caught
            int[] dr = { 0, 1, 0, -1 }; // array to store the four possible directions
            int[] dc = { 1, 0, -1, 0 };
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < m; j++)
                {
                    if (grid[i][j] > 0)
                    { // if the current cell contains fish
                        Queue<int[]> q = new Queue<int[]>(); // create a queue to perform BFS
                        q.Enqueue(new int[] { i, j }); // add the current cell to the queue
                        int f = grid[i][j]; // count the number of fish caught in the current cell
                        grid[i][j] = 0; // mark the current cell as visited by setting its value to 0
                        while (q.Count > 0)
                        { // while there are cells in the queue
                            int[] curr = q.Dequeue(); // remove the first cell from the queue
                            for (int k = 0; k < 4; k++)
                            { // iterate over the four possible directions
                                int nr = curr[0] + dr[k]; // calculate the coordinates of the adjacent cell
                                int nc = curr[1] + dc[k];
                                if (nr >= 0 && nr < n && nc >= 0 && nc < m && grid[nr][nc] > 0)
                                { // if the adjacent cell contains fish and is within the grid
                                    f += grid[nr][nc]; // count the number of fish caught in the adjacent cell
                                    grid[nr][nc] = 0; // mark the adjacent cell as visited by setting its value to 0
                                    q.Enqueue(new int[] { nr, nc }); // add the adjacent cell to the queue
                                }
                            }
                        }
                        maxFish = Math.Max(maxFish, f); // update the maximum number of fish caught so far
                    }
                }
            }
            return maxFish; // return the maximum number of fish caught
        }

        private int FindMaxFishDfs(int[][] grid)
        {
            int n = grid.Length;
            int m = grid[0].Length;
            int maxFish = 0; // variable to store the maximum number of fish caught
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < m; j++)
                {
                    if (grid[i][j] > 0)
                    { // if the current cell contains fish
                        maxFish = Math.Max(maxFish, FindMaxFishDfsRec(i, j, grid, n, m)); // update the maximum number of fish caught
                    }
                }
            }
            return maxFish; // return the maximum number of fish caught

        }

        // array to store the four possible directions
        private readonly int[] dr = { 0, 1, 0, -1, 0 };

        // function to perform DFS and count the number of fish caught
        private int FindMaxFishDfsRec(int i, int j, int[][] grid, int n, int m)
        {
            int fish = grid[i][j]; // count the number of fish caught in the current cell
            grid[i][j] = 0; // mark the current cell as visited by setting its value to 0
            for (int k = 0; k < 4; k++)
            { // iterate over the four possible directions
                int nr = i + dr[k], nc = j + dr[k + 1]; // calculate the coordinates of the adjacent cell
                if (nr < n && nr >= 0 && nc < m && nc >= 0 && grid[nr][nc] > 0)
                { // if the adjacent cell contains fish and is within the grid
                    fish += FindMaxFishDfsRec(nr, nc, grid, n, m); // count the number of fish caught in the adjacent cell
                }
            }
            return fish; // return the total number of fish caught

        }

        /*
        419. Battleships in a Board
        https://leetcode.com/problems/battleships-in-a-board/description/
        */
        public int CountBattleships(char[][] board)
        {
            int m = board.Length;
            if (m == 0) return 0;
            int n = board[0].Length;

            int count = 0;

            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    if (board[i][j] == '.') continue;
                    if (i > 0 && board[i - 1][j] == 'X') continue;
                    if (j > 0 && board[i][j - 1] == 'X') continue;
                    count++;
                }
            }

            return count;
        }
        /*
        1231. Divide Chocolate
        https://leetcode.com/problems/divide-chocolate/description/

        1. Using Binary Search
        Let n be the number of chunks in the chocolate and S be the total sweetness of the chocolate bar.
    	Time complexity: O(n⋅log(S/(k+1)))  The lower and upper bounds are min(sweetness) and S / (k + 1) respectively. 
                        In the worst case (when k is small), the right boundary will have the same magnitude as S, and the left boundary will be 1. 
                        Thus, the maximum possible time complexity for a single binary search is O(logS).
                        For every single search, we need to traverse the chocolate bar in order to allocate chocolate chunks to everyone, which takes O(n) time.
        Space complexity: O(1)For every search, we just need to count the number of people who receive a piece of chocolate with enough sweetness, and the total chocolate sweetness for the current people, both only cost constant space.

        */
        public int MaximizeSweetness(int[] sweetness, int k)
        {
            // Initialize the left and right boundaries.
            // left = 1 and right = total sweetness / number of people.
            int numberOfPeople = k + 1;
            int left = sweetness.Min();
            int right = sweetness.Sum() / numberOfPeople;

            while (left < right)
            {
                // Get the middle index between left and right boundary indexes.
                // cur_sweetness stands for the total sweetness for the current person.
                // people_with_chocolate stands for the number of people that have 
                // a piece of chocolate of sweetness greater than or equal to mid.  
                int mid = (left + right + 1) / 2;
                int curSweetness = 0;
                int peopleWithChocolate = 0;

                // Start assigning chunks to the current people,.
                foreach (int s in sweetness)
                {
                    curSweetness += s;

                    // If the total sweetness for him is no less than mid, meaning we 
                    // have done with him and should move on to assigning chunks to the next people.
                    if (curSweetness >= mid)
                    {
                        peopleWithChocolate += 1;
                        curSweetness = 0;
                    }
                }

                // Check if we successfully give everyone a piece of chocolate with sweetness
                // no less than mid, and eliminate the search space by half.
                if (peopleWithChocolate >= numberOfPeople)
                {
                    left = mid;
                }
                else
                {
                    right = mid - 1;
                }
            }

            // Once the left and right boundaries coincide, we find the target value,
            // that is, the maximum possible sweetness we can get.
            return right;
        }
        /*
        1011. Capacity To Ship Packages Within D Days
        https://leetcode.com/problems/capacity-to-ship-packages-within-d-days/description/

        Approach: Binary Search

        Here, n is the length of weights.
        Time complexity: O(n⋅log(500⋅n))=O(n⋅log(n)) It takes O(n) time to iterate through weights to compute maxLoad and totalLoad.
                            In the binary search algorithm, we divide our range by half every time. So for a range of length R, it performs O(log(R)) operations. 
                            In our case, the range is from maxLoad to totalLoad. As mentioned in the problem constraints, maxLoad can be 500, 
                            so the total load can be n * 500. So, in the worst case, the size of the range would be (n - 1) * 500 which would require O(log(500n−500))=O(log(n)) operations using a binary search algorithm.
                            To see if we can deliver the packages in the required number of days with a specific capacity, 
                            we iterate through the weights array to see if the current capacity allows us to carry the all the packages in days days, which needs O(n) time.
                            So it would take O(n⋅log(n)) time in total.
        Space complexity: O(1) We are only defining a few integer variables.

        */
        public int ShipWithinDays(int[] weights, int days)
        {
            int totalLoad = 0, maxLoad = 0;
            foreach (int weight in weights)
            {
                totalLoad += weight;
                maxLoad = Math.Max(maxLoad, weight);
            }

            int l = maxLoad, r = totalLoad;

            while (l < r)
            {
                int mid = (l + r) / 2;
                if (Feasible(weights, mid, days))
                {
                    r = mid;
                }
                else
                {
                    l = mid + 1;
                }
            }
            return l;
        }
        // Check whether the packages can be shipped in less than "days" days with
        // "c" capacity.
        Boolean Feasible(int[] weights, int c, int days)
        {
            int daysNeeded = 1, currentLoad = 0;
            foreach (int weight in weights)
            {
                currentLoad += weight;
                if (currentLoad > c)
                {
                    daysNeeded++;
                    currentLoad = weight;
                }
            }

            return daysNeeded <= days;
        }

    }
}
