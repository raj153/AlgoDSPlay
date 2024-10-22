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
using System.Reflection.Metadata.Ecma335;
using System.Text.RegularExpressions;
using AlgoDSPlay.Design;

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
        /*
        150. Evaluate Reverse Polish Notation
    https://leetcode.com/problems/evaluate-reverse-polish-notation/description/
    //https://www.algoexpert.io/questions/reversePolishNotation
        */
        public class EvalRevPolishNotationSol
        {
            /*
            Approach 1: Reducing the List In-place
Complexity Analysis
Let n be the length of the list.
•	Time Complexity : O(n^2).
Firstly, it helps to calculate how many operators and how many numbers are in the initial list. Each step of the algorithm removes 1 operator, 2 numbers, and adds back 1 number. This is an overall loss of 1 number and 1 operator per step. At the end, we have 1 number left. Therefore, we can infer that at the start, there must always be exactly 1 more number than there is operators.
The big inefficiency of this approach is more obvious in the Java code than the Python. Deleting an item from an ArrayList or Array is O(n), because all the items after have to be shuffled down one place to fill in the gap. The number of these deletions we need to do is the same as the number of operators, which is proportional to n. Therefore, the cost of the deletions is O(n2).
This is more obvious in the Java code, because we had to define the deletion method ourselves. However, the Python deletion method works the same way, it's just that you can't see it because it's hidden in a library function call. It's important to always be aware of the cost of library functions as they can sometimes look like they're O(1) when they're not!
•	Space Complexity : O(1).
The only extra space used is a constant number of single-value variables. Therefore, the overall algorithm requires O(1) space.
Interestingly, this approach could be adapted to work with a Double-Linked List. It would require O(n) space to create the list, and then take O(n) time to process it using a similar algorithm to above. This works because the algorithm is traversing the list in a linear fashion and modifications only impact the tokens immediately to the left of the current token.

            */
            private static Dictionary<string, Func<int, int, int>> OPERATIONS =
                new Dictionary<string, Func<int, int, int>>() {
            { "+", (int a, int b) => a + b },
            { "-", (int a, int b) => a - b },
            { "*", (int a, int b) => a * b },
            { "/", (int a, int b) => a / b }
                };

            public int EvalWithReduceInPlaceAlgo(string[] tokens)
            {
                int currentPosition = 0;
                while (tokens.Length > 1)
                {
                    while (!OPERATIONS.ContainsKey(tokens[currentPosition]))
                    {
                        currentPosition++;
                    }

                    string operation = tokens[currentPosition];
                    int number1 = Int32.Parse(tokens[currentPosition - 2]);
                    int number2 = Int32.Parse(tokens[currentPosition - 1]);
                    Func<int, int, int> func = OPERATIONS[operation];
                    int value = func(number1, number2);
                    tokens[currentPosition] = value.ToString();
                    List<string> tokenslist = tokens.ToList();
                    tokenslist.RemoveRange(currentPosition - 2, 2);
                    tokens = tokenslist.ToArray();
                    currentPosition--;
                }

                return Int32.Parse(tokens[0]);
            }

            /*
            Approach 2: Evaluate with Stack
Complexity Analysis
Let n be the length of the list.
•	Time Complexity : O(n).
We do a linear search to put all numbers on the stack, and process all operators. Processing an operator requires removing 2 numbers off the stack and replacing them with a single number, which is an O(1) operation. Therefore, the total cost is proportional to the length of the input array. Unlike before, we're no longer doing expensive deletes from the middle of an Array or List.
•	Space Complexity : O(n).
In the worst case, the stack will have all the numbers on it at the same time. This is never more than half the length of the input array.

            */
            public int EvalWithLambda(string[] tokens)
            {
                Stack<int> stack = new Stack<int>();
                foreach (string token in tokens)
                {
                    if (!OPERATIONS.ContainsKey(token))
                    {
                        stack.Push(Int32.Parse(token));
                        continue;
                    }

                    int number2 = stack.Pop();
                    int number1 = stack.Pop();
                    Func<int, int, int> operation = OPERATIONS[token];
                    int result = operation(number1, number2);
                    stack.Push(result);
                }

                return stack.Pop();
            }
            public static int EvalWithStack(string[] tokens)
            {
                Stack<int> stack = new Stack<int>();
                foreach (string token in tokens)
                {

                    if (!"+-*/".Contains(token))
                    {
                        stack.Push(Int32.Parse(token));
                        continue;
                    }

                    int number2 = stack.Pop();
                    int number1 = stack.Pop();
                    int result = 0;
                    switch (token)
                    {
                        case "+":
                            result = number1 + number2;
                            break;
                        case "-":
                            result = number1 - number2;
                            break;
                        case "*":
                            result = number1 * number2;
                            break;
                        case "/":
                            result = number1 / number2;
                            break;
                    }

                    stack.Push(result);
                }
                return stack.Pop();
            }
        }



        //https://www.algoexpert.io/questions/evaluate-expression-tree
        public int EvaluateExpressionTree(DataStructures.TreeNode tree)
        {
            //T:O(n) time | S: O(h) space - where n is the number of nodes in the Binary Tree, and h is the height of the Binary Tree
            return EvalExpTree(tree);
        }
        private int EvalExpTree(DataStructures.TreeNode node)
        {

            if (node.Val >= 0) return node.Val;

            int left = EvalExpTree(node.Left);
            int right = EvalExpTree(node.Right);
            int res = EvalExp(left, right, node.Val);

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
        /*
        84. Largest Rectangle in Histogram
https://leetcode.com/problems/largest-rectangle-in-histogram/description/

        */
        public class LargestRectangleAreaSolution
        {
            /*
Approach 1: Brute Force
Complexity Analysis
•	Time complexity: O(n^3). We have to find the minimum height bar O(n) lying
between every pair O(n^2).
•	Space complexity: O(1). Constant space is used.

*/
            public int LargestRectangleAreaNaive(int[] heights)
            {
                int max_area = 0;
                for (int i = 0; i < heights.Length; i++)
                {
                    for (int j = i; j < heights.Length; j++)
                    {
                        int min_height = Int32.MaxValue;
                        for (int k = i; k <= j; k++)
                        {
                            min_height = Math.Min(min_height, heights[k]);
                        }

                        max_area = Math.Max(max_area, min_height * (j - i + 1));
                    }
                }

                return max_area;
            }

            /*
            Approach 2: Better Brute Force

        Complexity Analysis
•	Time complexity: O(n^2). Every possible pair is considered
•	Space complexity: O(1). No extra space is used.
            */
            public int LargestRectangleAreaBetterNaive(int[] heights)
            {
                int maxArea = 0;
                int length = heights.Length;
                for (int i = 0; i < length; i++)
                {
                    int minHeight = int.MaxValue;
                    for (int j = i; j < length; j++)
                    {
                        minHeight = Math.Min(minHeight, heights[j]);
                        maxArea = Math.Max(maxArea, minHeight * (j - i + 1));
                    }
                }

                return maxArea;
            }

            /*
            Approach 3: Divide and Conquer Approach (DAC)
Complexity Analysis
•	Time complexity:
Average Case: O(nlogn).
Worst Case: O(n^2). If the numbers in the array are sorted, we don't gain the advantage of divide and conquer.
•	Space complexity: O(n). Recursion with worst case depth n.

            */
            public int LargestRectangleAreaDAC(int[] heights)
            {
                return CalculateArea(heights, 0, heights.Length - 1);
            }
            public int CalculateArea(int[] heights, int start, int end)
            {
                if (start > end)
                    return 0;
                int min_index = start;
                for (int i = start; i <= end; i++)
                    if (heights[min_index] > heights[i])
                        min_index = i;
                return Math.Max(heights[min_index] * (end - start + 1),
                                Math.Max(CalculateArea(heights, start, min_index - 1),
                                         CalculateArea(heights, min_index + 1, end)));
            }

            /*
            Approach 4: Better Divide and Conquer Using Segment Tree (DACST)

Complexity Analysis
•	Time complexity: O(nlogn). Segment tree takes logn for a total of n times.
•	Space complexity: O(n). Space required for Segment Tree.

            */

            public class LargestRectangleInHistogramSegmentTreeWithDAC
            {
                public int LargestRectangleArea(int[] heights)
                {
                    int n = heights.Length;
                    int[] segment = BuildSegmentTree(heights);
                    return DivideConquer(heights, 0, n - 1, segment);
                }

                private static int DivideConquer(int[] height, int l, int h, int[] segment)
                {
                    if (l <= h)
                    {
                        if (l == h) return height[l] * 1;
                        int minIndex = Query(segment, height, l, h);
                        int currArea = height[minIndex] * (h - l + 1);
                        int leftArea = DivideConquer(height, l, minIndex - 1, segment);
                        int rightArea = DivideConquer(height, minIndex + 1, h, segment);
                        return Math.Max(Math.Max(currArea, leftArea), rightArea);
                    }
                    return 0;
                }

                private static int[] BuildSegmentTree(int[] heights)
                {
                    int n = heights.Length;
                    int[] segment = new int[2 * n];
                    for (int i = n - 1, j = 2 * n - 1; i >= 0; i--, j--)
                    {
                        segment[j] = i;
                    }
                    for (int i = n - 1; i > 0; i--)
                    {
                        if (heights[segment[2 * i]] < heights[segment[2 * i + 1]]) segment[i] = segment[2 * i];
                        else segment[i] = segment[2 * i + 1];
                    }
                    return segment;
                }

                private static int Query(int[] segment, int[] heights, int i, int j)
                {
                    int n = heights.Length;
                    int p = i + n;
                    int q = j + n;
                    int min = int.MaxValue;
                    int index = -1;
                    while (p <= q)
                    {
                        if (p % 2 == 1)
                        {
                            if (heights[segment[p]] < min)
                            {
                                min = heights[segment[p]];
                                index = segment[p];
                            }
                            p++;
                        }
                        if (q % 2 == 0)
                        {
                            if (heights[segment[q]] < min)
                            {
                                min = heights[segment[q]];
                                index = segment[q];
                            }
                            q--;
                        }
                        p = p >> 1;
                        q = q >> 1;
                    }
                    return index;
                }



            }
            /*
         
            https://www.geeksforgeeks.org/largest-rectangular-area-in-a-histogram-using-segment-tree/
            */
            public class LargestRectangleInHistogramSegmentTreeWithoutDAC
            {

                static int[] hist;
                static int[] st;

                // A utility function to find minimum of three integers
                static int Max(int x, int y, int z)
                { return Math.Max(Math.Max(x, y), z); }

                // A utility function to get minimum of two numbers in hist[]
                static int MinVal(int i, int j)
                {
                    if (i == -1) return j;
                    if (j == -1) return i;
                    return (hist[i] < hist[j]) ? i : j;
                }

                // A utility function to get the middle index from corner indexes.
                static int GetMid(int s, int e)
                { return s + (e - s) / 2; }

                /* A recursive function to get the index of minimum value in a given range of
                    indexes. The following are parameters for this function.

                    hist -. Input array for which segment tree is built
                    st -. Pointer to segment tree
                    index -. Index of current node in the segment tree. Initially 0 is
                            passed as root is always at index 0
                    ss & se -. Starting and ending indexes of the segment represented by
                                current node, i.e., st[index]
                    qs & qe -. Starting and ending indexes of query range */
                static int RMQUtil(int ss, int se, int qs, int qe, int index)
                {
                    // If segment of this node is a part of given range, then return the
                    // min of the segment
                    if (qs <= ss && qe >= se)
                        return st[index];

                    // If segment of this node is outside the given range
                    if (se < qs || ss > qe)
                        return -1;

                    // If a part of this segment overlaps with the given range
                    int mid = GetMid(ss, se);
                    return MinVal(RMQUtil(ss, mid, qs, qe, 2 * index + 1),
                                RMQUtil(mid + 1, se, qs, qe, 2 * index + 2));
                }

                // Return index of minimum element in range from index qs (query start) to
                // qe (query end). It mainly uses RMQUtil()
                static int RMQ(int n, int qs, int qe)
                {
                    // Check for erroneous input values
                    if (qs < 0 || qe > n - 1 || qs > qe)
                    {
                        Console.Write("Invalid Input");
                        return -1;
                    }

                    return RMQUtil(0, n - 1, qs, qe, 0);
                }

                // A recursive function that constructs Segment Tree for hist[ss..se].
                // si is index of current node in segment tree st
                static int ConstructSTUtil(int ss, int se, int si)
                {
                    // If there is one element in array, store it in current node of
                    // segment tree and return
                    if (ss == se)
                        return (st[si] = ss);

                    // If there are more than one elements, then recur for left and
                    // right subtrees and store the minimum of two values in this node
                    int mid = GetMid(ss, se);
                    st[si] = MinVal(ConstructSTUtil(ss, mid, si * 2 + 1),
                                    ConstructSTUtil(mid + 1, se, si * 2 + 2));
                    return st[si];
                }

                /* Function to construct segment tree from given array. This function
                allocates memory for segment tree and calls constructSTUtil() to
                fill the allocated memory */
                static void ConstructST(int n)
                {
                    // Allocate memory for segment tree
                    int x = (int)(Math.Ceiling(Math.Log(n))); //Height of segment tree
                    int max_size = 2 * (int)Math.Pow(2, x) - 1; //Maximum size of segment tree
                    st = new int[max_size * 2];

                    // Fill the allocated memory st
                    ConstructSTUtil(0, n - 1, 0);

                    // Return the constructed segment tree
                    // return st;
                }

                // A recursive function to find the maximum rectangular area.
                // It uses segment tree 'st' to find the minimum value in hist[l..r]
                static int GetMaxAreaRec(int n, int l, int r)
                {
                    // Base cases
                    if (l > r) return Int32.MinValue;
                    if (l == r) return hist[l];

                    // Find index of the minimum value in given range
                    // This takes O(Logn)time
                    int m = RMQ(n, l, r);

                    /* Return maximum of following three possible cases
                    a) Maximum area in Left of min value (not including the min)
                    a) Maximum area in right of min value (not including the min)
                    c) Maximum area including min */
                    return Max(GetMaxAreaRec(n, l, m - 1),
                            GetMaxAreaRec(n, m + 1, r),
                            (r - l + 1) * (hist[m]));
                }

                // The main function to find max area
                static int GetMaxArea(int n)
                {
                    // Build segment tree from given array. This takes
                    // O(n) time
                    ConstructST(n);

                    // Use recursive utility function to find the
                    // maximum area
                    return GetMaxAreaRec(n, 0, n - 1);
                }

                // Driver Code
                public static void Main(string[] args)
                {
                    int[] a = { 6, 1, 5, 4, 5, 2, 6 };
                    int n = a.Length;
                    hist = new int[n];

                    hist = a;
                    Console.WriteLine("Maximum area is " + GetMaxArea(n));
                }
            }

            /*
                Approach 5: Using Stack
    Complexity Analysis
    •	Time complexity: O(n). n numbers are pushed and popped.
    •	Space complexity: O(n). Stack is used.

                */

            public int LargestRectangleAreaStack(int[] heights)
            {
                Stack<int> stack = new Stack<int>();
                stack.Push(-1);
                int maxArea = 0;
                for (int i = 0; i < heights.Length; i++)
                {
                    while (stack.Peek() != -1 && heights[stack.Peek()] >= heights[i])
                    {
                        int currentHeight = heights[stack.Pop()];
                        int currentWidth = i - stack.Peek() - 1;
                        maxArea = Math.Max(maxArea, currentHeight * currentWidth);
                    }

                    stack.Push(i);
                }

                while (stack.Peek() != -1)
                {
                    int currentHeight = heights[stack.Pop()];
                    int currentWidth = heights.Length - stack.Peek() - 1;
                    maxArea = Math.Max(maxArea, currentHeight * currentWidth);
                }

                return maxArea;
            }

        }

        /*
        85. Maximal Rectangle
        https://leetcode.com/problems/maximal-rectangle/description/
        */
        public class MaximalRectangleContainsOnlyOnesSol
        {
            /*
            Approach 1: Brute Force

            Complexity Analysis
•	Time complexity : O(N^3*M^3), with N being the number of rows and M the number of columns.
Iterating over all possible coordinates is O(N^2*M^2), and iterating over the rectangle defined by two coordinates is an additional O(NM). O(NM)∗O(N^2*M^2)=O(N^3*M^3).
•	Space complexity : O(1).

            */
            /*
        Approach 2: Dynamic Programming - Better Brute Force on Histograms 
        Complexity Analysis
•	Time complexity : O(N^2*M). Computing the maximum area for one point takes O(N) time, since it iterates over the values in the same column. This is done for all N∗M points, giving O(N)∗O(NM)=O(N^2*M).
•	Space complexity : O(NM). We allocate an equal sized array to store the maximum width at each point.
   
            */
            public int MaximalRectangleContainsOnlyOnesDP(char[][] matrix)
            {
                if (matrix.Length == 0)
                    return 0;
                int maxarea = 0;
                int[][] dp = new int[matrix.Length][];
                for (int a = 0; a < dp.Length; a++) dp[a] = new int[matrix[0].Length];
                for (int i = 0; i < matrix.Length; i++)
                {
                    for (int j = 0; j < matrix[0].Length; j++)
                    {
                        if (matrix[i][j] == '1')
                        {
                            // compute the maximum width and update dp with it
                            dp[i][j] = j == 0 ? 1 : dp[i][j - 1] + 1;
                            int width = dp[i][j];
                            // compute the maximum area rectangle with a lower right
                            // corner at [i, j]
                            for (int k = i; k >= 0; k--)
                            {
                                width = Math.Min(width, dp[k][j]);
                                maxarea = Math.Max(maxarea, width * (i - k + 1));
                            }
                        }
                    }
                }

                return maxarea;

            }
            /*
            Approach 3: Using Histograms - Stack
    Complexity Analysis
    •	Time complexity : O(NM). Running leetcode84 on each row takes M (length of each row) time. This is done N times for O(NM).
    •	Space complexity : O(M). We allocate an array the size of the the number of columns to store our widths at each row.

            */
            private int GetMaxRectOfOnesUsingStack(int[] heights)
            {
                Stack<int> stack = new Stack<int>();
                stack.Push(-1);
                int maxarea = 0;
                for (int i = 0; i < heights.Length; ++i)
                {
                    while (stack.Peek() != -1 && heights[stack.Peek()] >= heights[i])
                        maxarea = Math.Max(
                            maxarea, heights[stack.Pop()] * (i - stack.Peek() - 1));
                    stack.Push(i);
                }

                while (stack.Peek() != -1)
                    maxarea =
                        Math.Max(maxarea, heights[stack.Pop()] *
                                              (heights.Length - stack.Peek() - 1));
                return maxarea;
            }

            public int MaximalRectangleContainsOnlyOnesStack(char[][] matrix)
            {
                if (matrix.Length == 0)
                    return 0;
                int maxarea = 0;
                int[] dp = new int[matrix[0].Length];
                for (int i = 0; i < matrix.Length; i++)
                {
                    for (int j = 0; j < matrix[0].Length; j++)
                    {
                        dp[j] = matrix[i][j] == '1' ? dp[j] + 1 : 0;
                    }

                    maxarea = Math.Max(maxarea, GetMaxRectOfOnesUsingStack(dp));
                }

                return maxarea;
            }

            /*
            Approach 4: Dynamic Programming - Maximum Height at Each Point (DPMH)

            Complexity Analysis
    •	Time complexity : O(NM). In each iteration over N we iterate over M a constant number of times.
    •	Space complexity : O(M). M is the length of the additional arrays we keep.

            */
            public int MaximalRectangleContainsOnlyOnesDPMH(char[][] matrix)
            {
                if (matrix.Length == 0)
                    return 0;
                int m = matrix.Length;
                int n = matrix[0].Length;
                int[] left = new int[n];
                int[] right = new int[n];
                int[] height = new int[n];
                for (int i = 0; i < n; i++) right[i] = n;
                int maxarea = 0;
                for (int i = 0; i < m; i++)
                {
                    int cur_left = 0, cur_right = n;
                    for (int j = 0; j < n; j++)
                    {
                        if (matrix[i][j] == '1')
                            height[j]++;
                        else
                            height[j] = 0;
                    }

                    for (int j = 0; j < n; j++)
                    {
                        if (matrix[i][j] == '1')
                            left[j] = Math.Max(left[j], cur_left);
                        else
                        {
                            left[j] = 0;
                            cur_left = j + 1;
                        }
                    }

                    for (int j = n - 1; j >= 0; j--)
                    {
                        if (matrix[i][j] == '1')
                            right[j] = Math.Min(right[j], cur_right);
                        else
                        {
                            right[j] = n;
                            cur_right = j;
                        }
                    }

                    for (int j = 0; j < n; j++)
                    {
                        maxarea = Math.Max(maxarea, (right[j] - left[j]) * height[j]);
                    }
                }

                return maxarea;
            }

        }

        /*
        221. Maximal Square
        https://leetcode.com/problems/maximal-square/description/
        */
        public class MaximalSquareCotainsOnesSol
        {
            /*
            Approach 1: Brute Force
            Complexity Analysis
•	Time complexity : O((mn)^2). In worst case, we need to traverse the complete matrix for every 1.
•	Space complexity : O(1). No extra space is used.

            */
            public int MaximalSquareCotainsOnesNaive(char[][] matrix)
            {
                int rows = matrix.Length, cols = rows > 0 ? matrix[0].Length : 0;
                int maxsqlen = 0;
                for (int i = 0; i < rows; i++)
                {
                    for (int j = 0; j < cols; j++)
                    {
                        if (matrix[i][j] == '1')
                        {
                            int sqlen = 1;
                            bool flag = true;
                            while (sqlen + i < rows && sqlen + j < cols && flag)
                            {
                                for (int k = j; k <= sqlen + j; k++)
                                {
                                    if (matrix[i + sqlen][k] == '0')
                                    {
                                        flag = false;
                                        break;
                                    }
                                }
                                for (int k = i; k <= sqlen + i; k++)
                                {
                                    if (matrix[k][j + sqlen] == '0')
                                    {
                                        flag = false;
                                        break;
                                    }
                                }
                                if (flag) sqlen++;
                            }
                            if (maxsqlen < sqlen)
                            {
                                maxsqlen = sqlen;
                            }
                        }
                    }
                }
                return maxsqlen * maxsqlen;
            }
            /*
            Approach 2: Dynamic Programming (DP)
Complexity Analysis
•	Time complexity : O(mn). Single pass.
•	Space complexity : O(mn). Another matrix of same size is used for dp

            */
            public int MaximalSquareCotainsOnesDP(char[][] matrix)
            {
                int rows = matrix.Length, cols = rows > 0 ? matrix[0].Length : 0;
                int[][] dp = new int[rows + 1][];
                int maxsqlen = 0;
                // for convenience, we add an extra all zero column and row
                // outside of the actual dp table, to simpify the transition
                for (int i = 1; i <= rows; i++)
                {
                    for (int j = 1; j <= cols; j++)
                    {
                        if (matrix[i - 1][j - 1] == '1')
                        {
                            dp[i][j] = Math.Min(
                                Math.Min(dp[i][j - 1], dp[i - 1][j]),
                                dp[i - 1][j - 1]
                            ) +
                            1;
                            maxsqlen = Math.Max(maxsqlen, dp[i][j]);
                        }
                    }
                }
                return maxsqlen * maxsqlen;
            }

            /*        
    Approach 3: Better Dynamic Programming (BetrDP)
    Complexity Analysis
    •	Time complexity : O(mn). Single pass.
    •	Space complexity : O(n). Another array which stores elements in a row is used for dp

            */
            public int MaximalSquareCotainsOnesBetrDP(char[][] matrix)
            {
                int rows = matrix.Length, cols = rows > 0 ? matrix[0].Length : 0;
                int[] dp = new int[cols + 1];
                int maxsqlen = 0, prev = 0;
                for (int i = 1; i <= rows; i++)
                {
                    for (int j = 1; j <= cols; j++)
                    {
                        int temp = dp[j];
                        if (matrix[i - 1][j - 1] == '1')
                        {
                            dp[j] = Math.Min(Math.Min(dp[j - 1], prev), dp[j]) + 1;
                            maxsqlen = Math.Max(maxsqlen, dp[j]);
                        }
                        else
                        {
                            dp[j] = 0;
                        }
                        prev = temp;
                    }
                }
                return maxsqlen * maxsqlen;
            }

        }


        //https://www.algoexpert.io/questions/largest-park        

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
        /*
17. Letter Combinations of a Phone Number
https://leetcode.com/problems/letter-combinations-of-a-phone-number/description/

        */
        //Approach 1: Backtracking

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
                int documentFrequency = CountCharFrequency(character, document);
                int charactersFrequency = CountCharFrequency(character, characters);
                if (documentFrequency > charactersFrequency)
                {
                    return false;
                }
            }

            return true;
        }

        public int CountCharFrequency(char character, string target)
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

                int documentFrequency = CountCharFrequency(character, document);
                int charactersFrequency = CountCharFrequency(character, characters);
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
        1462. Course Schedule IV
        https://leetcode.com/problems/course-schedule-iv/description/
        https://algo.monster/liteproblems/1462
         */

        class CourseScheduleIVSol
        {
            /*Time and Space Complexity
Time Complexity
The given Python code performs multiple operations involving graphs and lists, so let's break down its time complexity:
1.	Initialization of f and g: The two lists are initialized with sizes n x n and n, respectively. Initializations run in O(n^2) for f and O(n) for g.
2.	Iterating over prerequisites list: The loop goes through the list of prerequisites, which in worst case will have O(n^2) elements (if every node is a prerequisite for every other node), and updates graph g and indeg. Each operation inside the loop takes constant time, so this step runs in O(E) where E is the number of edges or prerequisites.
3.	Processing the graph: The while loop dequeues nodes with indegree of 0 and processes adjacent nodes. Each edge is considered once when its destination node's indegree becomes 0, and the for loop inside processes n possible predecessors. This results in O(n + E) operations for the while loop since every node and edge is considered once. But due to the nested for loops, we're actually considering n times the adjacent nodes for updating f which yields O(n * E).
4.	Answering queries: This is a direct access operation for each query in f which takes O(1) time. There may be Q queries, thus this would take O(Q).
Combining all these, the overall time complexity is dominated by O(n * E), given that updating f can take up to O(n) time for each of the E edges in the worst-case scenario. Therefore, the time complexity is O(n^2 + E + n * E + Q) = O(n * E + Q).
Space Complexity
The space complexity of the algorithm can be analyzed as follows:
1.	Space for f: O(n^2) since it stores boolean values for every possible pair of nodes.
2.	Space for g and indeg: O(n) for each, since they store a list of adjacent nodes and indegree counts for each node, respectively.
3.	Additional space for the queue q: In the worst case, this can store all n nodes, so O(n).
So, the overall space complexity is the sum of these which gives O(n^2) + 2 * O(n) + O(n) = O(n^2) as n^2 will dominate the space complexity for large n.
  */
            public IList<bool> CheckIfPrerequisite(int numberOfCourses, int[][] prerequisites, int[][] queries)
            {
                // Floyd-Warshall algorithm to determine transitive closure of prerequisites
                bool[,] transitiveClosure = new bool[numberOfCourses, numberOfCourses];
                List<int>[] courseGraph = new List<int>[numberOfCourses];
                int[] inDegree = new int[numberOfCourses]; // For topological sorting

                // Initialize adjacency list
                for (int i = 0; i < numberOfCourses; i++)
                {
                    courseGraph[i] = new List<int>();
                }

                // Build graph and in-degree array from prerequisites
                foreach (int[] prerequisite in prerequisites)
                {
                    courseGraph[prerequisite[0]].Add(prerequisite[1]);
                    inDegree[prerequisite[1]]++; // Increment in-degree of successor
                }

                // Queue used for topological sorting
                Queue<int> queue = new Queue<int>();

                // Adding all nodes with in-degree 0 to queue
                for (int i = 0; i < numberOfCourses; i++)
                {
                    if (inDegree[i] == 0)
                    {
                        queue.Enqueue(i);
                    }
                }

                // Perform topological sort (Kahn's algorithm)
                while (queue.Count > 0)
                {
                    int currentCourse = queue.Dequeue();

                    // Explore all neighbors of the current course
                    foreach (int neighbor in courseGraph[currentCourse])
                    {
                        transitiveClosure[currentCourse, neighbor] = true;

                        // Update transitive closure for all nodes that lead to current
                        for (int preCourse = 0; preCourse < numberOfCourses; preCourse++)
                        {
                            transitiveClosure[preCourse, neighbor] |= transitiveClosure[preCourse, currentCourse];
                        }

                        // Decrement in-degree of neighbor and if 0, add to queue
                        if (--inDegree[neighbor] == 0)
                        {
                            queue.Enqueue(neighbor);
                        }
                    }
                }

                // Prepare the answer list to fulfill queries
                List<bool> answers = new List<bool>();

                // Check in the transitive closure if prerequisites are met
                foreach (int[] query in queries)
                {
                    answers.Add(transitiveClosure[query[0], query[1]]);
                }

                // Return the list of results for each query
                return answers;
            }
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
        https://algo.monster/liteproblems/2115

        */
        public List<string> FindAllRecipes(List<string> recipes, List<List<string>> ingredients, List<string> supplies)
        {
            /* Time and Space Complexity
Time Complexity
The time complexity of the findAllRecipes function is O(N + M + V) where:
•	N is the number of supplies.
•	M is the number of recipes.
•	V is the total number of ingredients across all recipes.
Here's why:
•	Building the graph with adjacency lists takes O(V) time, as you must process each ingredient of each recipe.
•	Filling the indegree map also takes O(V) time.
•	The breadth-first search (BFS) queue processing would be O(N + M) at most since each supply is put into the queue exactly once at the start, and then each recipe can only be put into the queue at most once when its indegree reaches zero.
•	Within the BFS, for each item, you look at its adjacency list and update indegrees. In total, across all BFS steps, you will examine and decrement each edge in the adjacency list only once. There are V edges since each edge corresponds to a unique ingredient in a recipe.
Space Complexity
The space complexity of the given code is O(N + M + V). The factors contributing to the space complexity include:
•	The queue can grow up to a maximum of the number of supplies plus recipes, O(N + M).
•	The graph g and indegree map indeg both combined will take up space proportional to the number of unique ingredients plus recipes, which is O(M + V).
 */
            // Initialize a graph to represent ingredients pointing to recipes
            Dictionary<string, List<string>> recipeGraph = new Dictionary<string, List<string>>();
            int recipeCount = recipes.Count;
            HashSet<string> supplySet = new HashSet<string>(supplies); // Store all the supplies in a hash set

            Dictionary<string, int> recipeIndegree = new Dictionary<string, int>(); // To store the indegree of all recipes
            foreach (var recipe in recipes)
            {
                recipeIndegree[recipe] = 0; // Initially set the indegree of all recipes to be 0
            }
            // Build the graph and indegree map
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
            // Queue to perform the topological sort
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
            //TreeMap<int, int> calendar;
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
                PriorityQueue<int> free = new PriorityQueue<>((a, b) -> a - b);
                PriorityQueue<Pair<int, int>> busy = new PriorityQueue<>((a, b) -> a.getKey() - b.getKey());
                
                // All servers are free at the beginning.

                for (int i = 0; i < k; ++i) {
                    free.add(i);
                }
                
                for (int i = 0; i < arrival.length; ++i) {
                    int start = arrival[i];

                    // Remove free servers from 'busy', modify their IDs and
                    // add them to 'free'
                    while (!busy.isEmpty() && busy.peek().getKey() <= start) {
                        Pair<int, int> curr = busy.remove();
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
                List<int> answer = new ArrayList<>();
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
        https://algo.monster/liteproblems/419
        */
        public int CountBattleships(char[][] board)
        {
            /* Time Complexity
We can check if a cell is a leader in O(1)O(1) and since there are O(MN)O(MN) cells, our time complexity is O(MN)O(MN).
Time Complexity: O(MN)O(MN)
Space Complexity
Since we only maintain a counter for the number of leaders, our space complexity is O(1)O(1).
Space Complexity: O(1)O(1)	
 */
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
        /*
        2288. Apply Discount to Prices
        https://leetcode.com/problems/apply-discount-to-prices/description/

        */
        public string DiscountPrices(string sentence, int discount)
        {
            string[] words = sentence.Split(' ');

            for (int index = 0; index < words.Length; ++index)
            {
                string word = words[index];

                if (word.Length < 2) continue;

                char firstchar = word[0];
                string numberString = word.Substring(1);

                if (IsValid(firstchar, numberString))
                {
                    words[index] = Format(numberString, discount);
                }
            }
            return string.Join(" ", words);
        }

        private bool IsValid(char firstchar, string numberString)
        {
            if (firstchar != '$') return false;

            foreach (char character in numberString)
            {
                if (character < '0' || character > '9') return false;
            }
            return true;
        }

        private string Format(string numberString, int discount)
        {
            double discountedPrice = long.Parse(numberString) * (1 - discount / 100d);
            return '$' + discountedPrice.ToString("F2");
        }
        /*
        11. Container With Most Water
        https://leetcode.com/problems/container-with-most-water/
        */
        public int MaxWaterArea(int[] height)
        {
            /*
            Approach 1: Brute Force
Complexity Analysis
•	Time complexity: O(n^2). Calculating area for all (n(n−1))/2 height pairs.
•	Space complexity: O(1). Constant extra space is used

            */
            int maxWaterArea = MaxWaterAreaNaive(height);
            /*
Approach 2: Two Pointer Approach
Complexity Analysis
•	Time complexity: O(n). Single pass.
•	Space complexity: O(1). Constant space is used.
            */
            maxWaterArea = MaxWaterxAreaOptimal(height);

            return maxWaterArea;

        }
        private int MaxWaterAreaNaive(int[] height)
        {
            int maxarea = 0;

            for (int left = 0; left < height.Length; left++)
            {
                for (int right = left + 1; right < height.Length; right++)
                {
                    int width = right - left;
                    maxarea = Math.Max(
                        maxarea, Math.Min(height[left], height[right]) * width);
                }
            }

            return maxarea;
        }

        private int MaxWaterxAreaOptimal(int[] height)
        {
            int maxArea = 0;
            int left = 0;
            int right = height.Length - 1;

            while (left < right)
            {
                int width = right - left;
                maxArea = Math.Max(maxArea,
                                   Math.Min(height[left], height[right]) * width);
                if (height[left] <= height[right])
                {
                    left++;
                }
                else
                {
                    right--;
                }
            }

            return maxArea;
        }

        /*
42. Trapping Rain Water	
https://leetcode.com/problems/trapping-rain-water/description/
        */
        public int TrapRainWater(int[] height)
        {
            /*
            Approach 1: Brute Force
Complexity Analysis
•	Time complexity: O(n^2). For each element of array, we iterate the left and right parts.
•	Space complexity: O(1) extra space.
            */
            int trapRainWater = TrapRainWaterNaive(height);

            /*
Approach 2: Dynamic Programming
 Complexity Analysis
•	Time complexity: O(n).
o	We store the maximum heights upto a point using 2 iterations of O(n) each.
o	We finally update ans using the stored values in O(n).
•	Space complexity: O(n) extra space.
o	Additional O(n) space for left_max and right_max arrays than in Approach 1.
            
            */
            trapRainWater = TrapRainWaterDP(height);

            /*
   Approach 3: Using stacks         
Complexity Analysis
•	Time complexity: O(n).
o	Single iteration of O(n) in which each bar can be touched at most twice(due to insertion and deletion from stack) and insertion and deletion from stack takes O(1) time.
•	Space complexity: O(n). Stack can take upto O(n) space in case of stairs-like or flat structure.
       
            */
            trapRainWater = TrapRainWaterStack(height);
            /*
  Approach 4: Using 2 pointers          
   Complexity Analysis
•	Time complexity: O(n). Single iteration of O(n).
•	Space complexity: O(1) extra space. Only constant space required for left, right, left_max and right_max.


            */
            trapRainWater = TrapRainWaterOptimal(height);

            return trapRainWater;

        }

        public int TrapRainWaterNaive(int[] height)
        {
            int ans = 0;
            int size = height.Length;
            for (int i = 1; i < size - 1; i++)
            {
                int left_max = 0, right_max = 0;
                // Search the left part for max bar size
                for (int j = i; j >= 0; j--)
                {
                    left_max = Math.Max(left_max, height[j]);
                }
                // Search the right part for max bar size
                for (int j = i; j < size; j++)
                {
                    right_max = Math.Max(right_max, height[j]);
                }
                ans += Math.Min(left_max, right_max) - height[i];
            }
            return ans;
        }
        public int TrapRainWaterDP(int[] height)
        {
            // Case of empty height array
            if (height.Length == 0)
                return 0;
            int ans = 0;
            int size = height.Length;
            // Create left and right max arrays
            int[] left_max = new int[size];
            int[] right_max = new int[size];
            // Initialize first height into left max
            left_max[0] = height[0];
            for (int i = 1; i < size; i++)
            {
                // update left max with current max
                left_max[i] = Math.Max(height[i], left_max[i - 1]);
            }

            // Initialize last height into right max
            right_max[size - 1] = height[size - 1];
            for (int i = size - 2; i >= 0; i--)
            {
                // update right max with current max
                right_max[i] = Math.Max(height[i], right_max[i + 1]);
            }

            // Calculate the trapped water
            for (int i = 1; i < size - 1; i++)
            {
                ans += Math.Min(left_max[i], right_max[i]) - height[i];
            }

            // Return amount of trapped water
            return ans;
        }
        public int TrapRainWaterStack(int[] height)
        {
            int ans = 0, current = 0;
            Stack<int> st = new Stack<int>();
            while (current < height.Length)
            {
                while (st.Count != 0 && height[current] > height[st.Peek()])
                {
                    int top = st.Peek();
                    st.Pop();
                    if (st.Count == 0)
                        break;
                    int distance = current - st.Peek() - 1;
                    int bounded_height =
                        Math.Min(height[current], height[st.Peek()]) - height[top];
                    ans += distance * bounded_height;
                }

                st.Push(current++);
            }

            return ans;
        }
        public int TrapRainWaterOptimal(int[] height)
        {
            int left = 0, right = height.Length - 1;
            int ans = 0;
            int left_max = 0, right_max = 0;
            while (left < right)
            {
                if (height[left] < height[right])
                {
                    left_max = Math.Max(left_max, height[left]);
                    ans += left_max - height[left];
                    ++left;
                }
                else
                {
                    right_max = Math.Max(right_max, height[right]);
                    ans += right_max - height[right];
                    --right;
                }
            }

            return ans;
        }
        /*
        407. Trapping Rain Water II
        https://leetcode.com/problems/trapping-rain-water-ii/description/

        Using Proirity Queues
        Complexity
            Time complexity:
            O(mn log(mn))

        */
        class Cell
        {
            public int row;
            public int col;
            public int height;
            public Cell(int _row, int _col, int _height)
            {
                row = _row;
                col = _col;
                height = _height;
            }
        }
        public int TrapRainWater(int[][] heightMap)
        {
            int m = heightMap.Length, n = heightMap[0].Length;
            PriorityQueue<Cell, int> pq = new(Comparer<int>.Create((a, b) => a.CompareTo(b)));
            // Initially, add all the Cells which are on borders to the queue.
            for (int r = 0; r < m; r++)
            {
                pq.Enqueue(new Cell(r, 0, heightMap[r][0]), heightMap[r][0]);
                pq.Enqueue(new Cell(r, n - 1, heightMap[r][n - 1]), heightMap[r][n - 1]);
                heightMap[r][n - 1] = -1;
                heightMap[r][0] = -1;
            }
            for (int c = 1; c < n - 1; c++)
            {
                pq.Enqueue(new Cell(0, c, heightMap[0][c]), heightMap[0][c]);
                pq.Enqueue(new Cell(m - 1, c, heightMap[m - 1][c]), heightMap[m - 1][c]);
                heightMap[0][c] = -1;
                heightMap[m - 1][c] = -1;
            }


            // from the borders, pick the shortest cell visited and check its neighbors:
            // if the neighbor is shorter, collect the water it can trap and update its height as its height plus the water trapped
            // add all its neighbors to the queue.

            Tuple<int, int>[] dir = new Tuple<int, int>[]
            {Tuple.Create(0,1), Tuple.Create(0,-1),Tuple.Create(1,0),Tuple.Create(-1,0)};
            int res = 0;
            while (pq.Count > 0)
            {
                var cell = pq.Dequeue();
                for (int i = 0; i < dir.Length; i++)
                {
                    int row = cell.row + dir[i].Item1;
                    int col = cell.col + dir[i].Item2;

                    if (row < 0 || row >= m || col < 0 || col >= n || heightMap[row][col] == -1) continue;

                    res += Math.Max(0, cell.height - heightMap[row][col]);
                    Cell newBoundary = new Cell(row, col, Math.Max(heightMap[row][col], cell.height));
                    pq.Enqueue(newBoundary, newBoundary.height);
                    heightMap[row][col] = -1;
                }
            }

            return res;
        }

        /*
        1326. Minimum Number of Taps to Open to Water a Garden
        https://leetcode.com/problems/minimum-number-of-taps-to-open-to-water-a-garden/description
        */

        public int MinTaps(int n, int[] ranges)
        {

            /*
Approach 1: Dynamic Programming 
Complexity Analysis
Let m be the average range of the taps.
•	Time Complexity: O(n⋅m).
Iterating through each tap and updating the minimum number of taps for each position within its range requires nested loops. The outer loop iterates through each of the n+1 taps. The inner loop iterates through the positions within the range of each tap. The number of iterations for the inner loop is O(m).
Overall, the time complexity of the solution is O(n⋅m).
•	Space Complexity: O(n).
The space complexity is determined by the additional memory used to store the DP array. The size of the DP array is n+1.
Therefore, the space complexity is O(n).
            */
            int minTaps = MinTapsDP(n, ranges);
            /*
Approach 2: Greedy
Complexity Analysis
Time Complexity: O(n).
We iterate through the garden once to calculate the maximum reach for each position, and then iterate through the garden again to choose the taps and determine the minimum number of taps required. The iteration involves visiting each position in the garden once, resulting in a linear time complexity.

Space Complexity: O(n).
We use additional space to store the max_reach array of size n+1. Therefore, the space complexity is linear with respect to the size of the garden.                   
            */
            minTaps = MinTapsGreedy(n, ranges);

            /*
            Approach3: using Priority Queue

            Time complexity:
            O(n log n)
            */
            minTaps = MinTapsPQ(n, ranges);

            return minTaps;

        }
        public int MinTapsPQ(int n, int[] ranges)
        {
            PriorityQueue<int[], int> pq = new(Comparer<int>.Create((a, b) => a.CompareTo(b)));
            for (int i = 0; i < ranges.Length; i++)
            {
                int start = Math.Max(0, i - ranges[i]);
                int end = Math.Min(n, i + ranges[i]);
                if (start < end) pq.Enqueue(new int[] { start, end }, start);
            }
            int covered = 0, res = 0;
            while (pq.Count > 0 && covered < n)
            {
                int end = 0;
                while (pq.Count > 0 && pq.Peek()[0] <= covered)
                    end = Math.Max(end, pq.Dequeue()[1]);
                res++;
                if (end == covered) return -1;
                covered = end;
            }
            if (covered == n)
                return res;
            return -1;
        }

        public int MinTapsDP(int gardenLength, int[] ranges)
        {
            // Define an infinite value
            const int InfiniteValue = (int)1e9;

            // Create an array to store the minimum number of taps needed for each position
            int[] minimumTaps = new int[gardenLength + 1];
            Array.Fill(minimumTaps, InfiniteValue);

            // Initialize the starting position of the garden
            minimumTaps[0] = 0;

            for (int i = 0; i <= gardenLength; i++)
            {
                // Calculate the leftmost position reachable by the current tap
                int tapStart = Math.Max(0, i - ranges[i]);
                // Calculate the rightmost position reachable by the current tap
                int tapEnd = Math.Min(gardenLength, i + ranges[i]);

                for (int j = tapStart; j <= tapEnd; j++)
                {
                    // Update with the minimum number of taps
                    minimumTaps[tapEnd] = Math.Min(minimumTaps[tapEnd], minimumTaps[j] + 1);
                }
            }

            // Check if the garden can be watered completely
            if (minimumTaps[gardenLength] == InfiniteValue)
            {
                // Garden cannot be watered
                return -1;
            }

            // Return the minimum number of taps needed to water the entire garden
            return minimumTaps[gardenLength];
        }
        public int MinTapsGreedy(int gardenLength, int[] tapRanges)
        {
            // Create an array to track the maximum reach for each position
            int[] maximumReach = new int[gardenLength + 1];

            // Calculate the maximum reach for each tap
            for (int tapIndex = 0; tapIndex < tapRanges.Length; tapIndex++)
            {
                // Calculate the leftmost position the tap can reach
                int startPosition = Math.Max(0, tapIndex - tapRanges[tapIndex]);
                // Calculate the rightmost position the tap can reach
                int endPosition = Math.Min(gardenLength, tapIndex + tapRanges[tapIndex]);

                // Update the maximum reach for the leftmost position
                maximumReach[startPosition] = Math.Max(maximumReach[startPosition], endPosition);
            }

            // Number of taps used
            int numberOfTaps = 0;
            // Current rightmost position reached
            int currentEndPosition = 0;
            // Next rightmost position that can be reached
            int nextEndPosition = 0;

            // Iterate through the garden
            for (int position = 0; position <= gardenLength; position++)
            {
                // Current position cannot be reached
                if (position > nextEndPosition)
                {
                    return -1;
                }

                // Increment taps when moving to a new tap
                if (position > currentEndPosition)
                {
                    numberOfTaps++;
                    // Move to the rightmost position that can be reached
                    currentEndPosition = nextEndPosition;
                }

                // Update the next rightmost position that can be reached
                nextEndPosition = Math.Max(nextEndPosition, maximumReach[position]);
            }

            // Return the minimum number of taps used
            return numberOfTaps;
        }
        /*
        198. House Robber
        https://leetcode.com/problems/house-robber/
        */
        public int Rob(int[] nums)
        {
            /*
Approach 1: Recursion with Memoization
Complexity Analysis
•	Time Complexity: O(N) since we process at most N recursive calls, thanks to caching, and during each of these calls, we make an O(1) computation which is simply making two other recursive calls, finding their maximum, and populating the cache based on that.
•	Space Complexity: O(N) which is occupied by the cache and also by the recursion stack.
            
            */
            int maxMoneyRobbed = RobRecMemo(nums);
            /*
Approach 2: Dynamic Programming
Complexity Analysis
•	Time Complexity: O(N) since we have a loop from N−2⋯0 and we simply use the pre-calculated values of our dynamic programming table for calculating the current value in the table which is a constant time operation.
•	Space Complexity: O(N) which is used by the table. So what is the real advantage of this solution over the previous solution? In this case, we don't have a recursion stack. When the number of houses is large, a recursion stack can become a serious limitation, because the recursion stack size will be huge and the compiler will eventually run into stack-overflow problems (no pun intended!).
            
            */
            maxMoneyRobbed = RobDP(nums);


            /*
Approach 3: Optimized Dynamic Programming
Complexity Analysis
•	Time Complexity: O(N) since we have a loop from N−2⋯0 and we use the precalculated values of our dynamic programming table to calculate the current value in the table which is a constant time operation.
•	Space Complexity: O(1) since we are not using a table to store our values. Simply using two variables will suffice for our calculations.
            
            */
            maxMoneyRobbed = RobDPOptimal(nums);

            return maxMoneyRobbed;


        }
        public int RobDPOptimal(int[] nums)
        {
            int N = nums.Length;

            // Special handling for empty array case.
            if (N == 0)
            {
                return 0;
            }

            int robNext, robNextPlusOne;

            // Base case initializations.
            robNextPlusOne = 0;
            robNext = nums[N - 1];

            // DP table calculations. Note: we are not using any
            // table here for storing values. Just using two
            // variables will suffice.
            for (int i = N - 2; i >= 0; --i)
            {
                // Same as the recursive solution.
                int current = Math.Max(robNext, robNextPlusOne + nums[i]);

                // Update the variables
                robNextPlusOne = robNext;
                robNext = current;
            }

            return robNext;
        }
        public int RobDP(int[] nums)
        {
            int N = nums.Length;

            // Special handling for empty array case.
            if (N == 0)
            {
                return 0;
            }

            int[] maxRobbedAmount = new int[nums.Length + 1];

            // Base case initializations.
            maxRobbedAmount[N] = 0;
            maxRobbedAmount[N - 1] = nums[N - 1];

            // DP table calculations.
            for (int i = N - 2; i >= 0; --i)
            {
                // Same as the recursive solution.
                maxRobbedAmount[i] = Math.Max(
                    maxRobbedAmount[i + 1],
                    maxRobbedAmount[i + 2] + nums[i]
                );
            }

            return maxRobbedAmount[0];
        }
        public int RobRecMemo(int[] houseValues)
        {
            this.memo = new int[100];

            // Fill with sentinel value representing not-calculated recursions.
            Array.Fill(this.memo, -1);

            return this.RobFrom(0, houseValues);
        }

        private int RobFrom(int index, int[] houseValues)
        {
            // No more houses left to examine.
            if (index >= houseValues.Length)
            {
                return 0;
            }

            // Return cached value.
            if (this.memo[index] > -1)
            {
                return this.memo[index];
            }

            // Recursive relation evaluation to get the optimal answer.
            int optimalAmount = Math.Max(
                this.RobFrom(index + 1, houseValues),
                this.RobFrom(index + 2, houseValues) + houseValues[index]
            );

            // Cache for future use.
            this.memo[index] = optimalAmount;
            return optimalAmount;
        }
        /*
        213. House Robber II
        https://leetcode.com/problems/house-robber-ii/description/

        */
        public int RobII(int[] nums)
        {
            /*   
Approach 1: Dynamic Programming
Complexity Analysis
•	Time complexity : O(N) where N is the size of nums. We are accumulating results as we are scanning nums.
•	Space complexity : O(1) since we are not consuming additional space other than variables for two previous results and a temporary variable to hold one of the previous results.
            
            */

            if (nums.Length == 0) return 0;

            if (nums.Length == 1) return nums[0];

            int max1 = RobSimple(nums, 0, nums.Length - 2);
            int max2 = RobSimple(nums, 1, nums.Length - 1);

            return Math.Max(max1, max2);


        }
        public int RobSimple(int[] nums, int start, int end)
        {
            int t1 = 0;
            int t2 = 0;

            for (int i = start; i <= end; i++)
            {
                int temp = t1;
                int current = nums[i];
                t1 = Math.Max(current + t2, t1);
                t2 = temp;
            }

            return t1;
        }

        public class TreeNode
        {
            public int Val;
            public TreeNode Left;
            public TreeNode Right;
            public TreeNode(int val = 0, TreeNode left = null, TreeNode right = null)
            {
                this.Val = val;
                this.Left = left;
                this.Right = right;
            }
        }
        /*
337. House Robber III
https://leetcode.com/problems/house-robber-iii/description/

        */
        public int RobIII(TreeNode root)
        {
            /*
Approach 1: Recursion
Complexity Analysis
Let N be the number of nodes in the binary tree.
•	Time complexity: O(N) since we visit all nodes once.
•	Space complexity: O(N) since we need stacks to do recursion, and the maximum depth of the recursion is the height of the tree, which is O(N) in the worst case and O(log(N)) in the best case.

            
            */

            int result = RobIIIRec(root);

            /*
 Approach 2: Recursion with Memoization
Complexity Analysis
Let N be the number of nodes in the binary tree.
•	Time complexity: O(N) since we run the helper function for all nodes once, and saved the results to prevent the second calculation.
•	Space complexity: O(N) since we need two maps with the size of O(N) to store the results, and O(N) space for stacks to start recursion.

            
            */
            result = RobIIIRecMemo(root);
            /*
Approach 3: Dynamic Programming
Complexity Analysis
Let N be the number of nodes in the binary tree.
•	Time complexity: O(N) since we visit all nodes once to form the tree-array, and then iterate two DP array, which both have length O(N).
•	Space complexity: O(N) since we need an array of length O(N) to store the tree, and two DP arrays of length O(N). Also, the sizes of other data structures in code do not exceed O(N).

            
            */
            result = RobIIIDP(root);

            return result;


        }
        public int[] Helper(TreeNode node)
        {
            // return [rob this node, not rob this node]
            if (node == null)
            {
                return new int[] { 0, 0 };
            }
            int[] left = Helper(node.Left);
            int[] right = Helper(node.Right);
            // if we rob this node, we cannot rob its children
            int rob = node.Val + left[1] + right[1];
            // else, we free to choose rob its children or not
            int notRob = Math.Max(left[0], left[1]) + Math.Max(right[0], right[1]);

            return new int[] { rob, notRob };
        }

        public int RobIIIRec(TreeNode root)
        {
            int[] answer = Helper(root);
            return Math.Max(answer[0], answer[1]);
        }

        private Dictionary<TreeNode, int> robResult = new Dictionary<TreeNode, int>();
        private Dictionary<TreeNode, int> notRobResult = new Dictionary<TreeNode, int>();

        public int Helper(TreeNode node, bool parentRobbed)
        {
            if (node == null)
            {
                return 0;
            }
            if (parentRobbed)
            {
                if (robResult.ContainsKey(node))
                {
                    return robResult[node];
                }
                int result = Helper(node.Left, false) + Helper(node.Right, false);
                robResult[node] = result;
                return result;
            }
            else
            {
                if (notRobResult.ContainsKey(node))
                {
                    return notRobResult[node];
                }
                int rob = node.Val + Helper(node.Left, true) + Helper(node.Right, true);
                int notRob = Helper(node.Left, false) + Helper(node.Right, false);
                int result = Math.Max(rob, notRob);
                notRobResult[node] = result;
                return result;
            }
        }

        public int RobIIIRecMemo(TreeNode root)
        {
            return Helper(root, false);
        }

        public int RobIIIDP(TreeNode root)
        {
            if (root == null)
            {
                return 0;
            }
            // reform tree into array-based tree
            List<int> tree = new List<int>();
            Dictionary<int, List<int>> graph = new Dictionary<int, List<int>>();
            graph.Add(-1, new List<int>());
            int index = -1;
            // we use two Queue to store node and index
            Queue<TreeNode> q_node = new Queue<TreeNode>();
            q_node.Enqueue(root);
            Queue<int> q_index = new Queue<int>();
            q_index.Enqueue(index);

            while (q_node.Count > 0)
            {
                TreeNode node = q_node.Dequeue();
                int parentIndex = q_index.Dequeue();
                if (node != null)
                {
                    index++;
                    tree.Add(node.Val);
                    graph.Add(index, new List<int>());
                    graph[parentIndex].Add(index);
                    // push new node into Queue
                    q_node.Enqueue(node.Left);
                    q_index.Enqueue(index);
                    q_node.Enqueue(node.Right);
                    q_index.Enqueue(index);
                }
            }

            // represent the maximum start by node i with robbing i
            int[] dpRob = new int[index + 1];

            // represent the maximum start by node i without robbing i
            int[] dpNotRob = new int[index + 1];

            for (int i = index; i >= 0; i--)
            {
                List<int> children = graph[i];
                if (children == null || children.Count == 0)
                {
                    // if is leaf
                    dpRob[i] = tree[i];
                    dpNotRob[i] = 0;
                }
                else
                {
                    dpRob[i] = tree[i];
                    foreach (int child in children)
                    {
                        dpRob[i] += dpNotRob[child];
                        dpNotRob[i] += Math.Max(dpRob[child], dpNotRob[child]);
                    }
                }
            }

            return Math.Max(dpRob[0], dpNotRob[0]);
        }

        /* 2560. House Robber IV
        https://leetcode.com/problems/house-robber-iv/description/
        https://algo.monster/liteproblems/2560
         */
        class HouseRobberIVSol
        {

            /* Time Complexity
            The given Python function minCapability uses binary search through the bisect_left function to find the minimum capability required. It applies a helper function f as a key which processes the nums list on each iteration.
            •	The bisect_left function performs a binary search over a range of size max(nums) + 1, which involves O(log(Max)) iterations, where Max is the maximum element in nums.
            •	Within each binary search iteration, the helper function f performs a linear scan over the nums list of size n, to check how many elements can be skipped without exceeding capability x. The time complexity for this will be O(n).
            Combining these two, the overall time complexity will be O(n * log(Max)), where n is the number of elements in nums, and Max is the maximum element in nums.
            Space Complexity
            The space complexity of the minCapability function is O(1) assuming that the list nums is given and does not count towards the space complexity (as it's an input). There are no additional data structures used that grow with the input size. The variables cnt, j, i, and v use a constant amount of space.
             */
            // Determine the minimum capability to partition the array in such a way that
            // the sum of each sub-array is less than or equal to k
            public int MinCapability(int[] nums, int k)
            {
                // Start with the least possible capability
                int left = 0;
                // Set an upper limit for the search space, assuming the max value according to problem constraints
                int right = (int)1e9;

                // Perform a binary search to find the minimum capability
                while (left < right)
                {
                    // Get the midpoint of the current search space
                    int mid = (left + right) >> 1;

                    // Check if the current capability can achieve the required partition
                    if (CalculatePartitionCount(nums, mid) >= k)
                    {
                        // If it qualifies, search the lower half to find a smaller capability
                        right = mid;
                    }
                    else
                    {
                        // Otherwise, search the upper half
                        left = mid + 1;
                    }
                }

                // left is now the minimum capability that can achieve the required partition
                return left;
            }

            // Helper method to calculate the number of partitions formed by capability x
            private int CalculatePartitionCount(int[] nums, int x)
            {
                int count = 0; // Initialize the partition count
                int lastPartitionIndex = -2; // Initialize the index of the last partition start

                // Iterate over the array
                for (int i = 0; i < nums.Length; ++i)
                {
                    // Skip if the current number exceeds the capability or is the next immediate number after the last partition
                    if (nums[i] > x || i == lastPartitionIndex + 1)
                    {
                        continue;
                    }
                    // Increment the partition count and update lastPartitionIndex
                    ++count;
                    lastPartitionIndex = i;
                }

                // Return the total number of partitions that can be made with capability x
                return count;
            }
        }


        /*
        656. Coin Path
https://leetcode.com/problems/coin-path/description/	

        */
        public List<int> CheapestJump(int[] coins, int maxJump)
        {
            /*
  Approach #1 Brute Force[Time Limit Exceeded]
  Complexity Analysis
  •	Time complexity : O(B^n). The size of the recursive tree can grow upto O(b^n) in the worst case. This is because, we have B possible branches at every step. Here, B refers to the limit of the largest jump and n refers to the size of the given A array.
  •	Space complexity : O(n). The depth of the recursive tree can grow upto n. next array of size n is used.

            */
            var cheapestJump = CheapestJumpNaive(coins, maxJump);
            /*
  Approach #2 Using Memoization 
  Complexity Analysis
  •	Time complexity : O(nB). memo array of size n is filled only once. We also do a traversal over the next array, which will go upto B steps. Here, n refers to the number of nodes in the given tree.
  •	Space complexity : O(n). The depth of the recursive tree can grow upto n. next array of size n is used.

            */
            cheapestJump = CheapestJumpMemo(coins, maxJump);
            /*
 Approach #3 Using Dynamic Programming 
 Complexity Analysis
 •	Time complexity : O(nB). We need to consider all the possible B positions for every current index considered in the A array. Here, A refers to the number of elements in A.
 •	Space complexity : O(n). dp and next array of size n are used.

           */
            cheapestJump = CheapestJumpDP(coins, maxJump);

            return cheapestJump;

        }
        public List<int> CheapestJumpDP(int[] costs, int maxJump)
        {
            int[] nextIndex = new int[costs.Length];
            long[] minimumCost = new long[costs.Length];
            Array.Fill(nextIndex, -1);
            List<int> result = new List<int>();

            for (int currentIndex = costs.Length - 2; currentIndex >= 0; currentIndex--)
            {
                long minCost = int.MaxValue;

                for (int jumpIndex = currentIndex + 1; jumpIndex <= currentIndex + maxJump && jumpIndex < costs.Length; jumpIndex++)
                {
                    if (costs[jumpIndex] >= 0)
                    {
                        long totalCost = costs[currentIndex] + minimumCost[jumpIndex];
                        if (totalCost < minCost)
                        {
                            minCost = totalCost;
                            nextIndex[currentIndex] = jumpIndex;
                        }
                    }
                }
                minimumCost[currentIndex] = minCost;
            }

            int index;
            for (index = 0; index < costs.Length && nextIndex[index] > 0; index = nextIndex[index])
                result.Add(index + 1);

            if (index == costs.Length - 1 && costs[index] >= 0)
                result.Add(costs.Length);
            else
                return new List<int>();

            return result;
        }
        public List<int> CheapestJumpMemo(int[] array, int jumpDistance)
        {
            int[] nextIndices = new int[array.Length];
            Array.Fill(nextIndices, -1);
            long[] memoization = new long[array.Length];
            Jump(array, jumpDistance, 0, nextIndices, memoization);
            List<int> result = new List<int>();
            int index;
            for (index = 0; index < array.Length && nextIndices[index] > 0; index = nextIndices[index])
            {
                result.Add(index + 1);
            }
            if (index == array.Length - 1 && array[index] >= 0)
            {
                result.Add(array.Length);
            }
            else
            {
                return new List<int>();
            }
            return result;
        }

        public long Jump(int[] array, int jumpDistance, int currentIndex, int[] nextIndices, long[] memoization)
        {
            if (memoization[currentIndex] > 0)
            {
                return memoization[currentIndex];
            }
            if (currentIndex == array.Length - 1 && array[currentIndex] >= 0)
            {
                return array[currentIndex];
            }
            long minimumCost = int.MaxValue;
            for (int nextIndex = currentIndex + 1; nextIndex <= currentIndex + jumpDistance && nextIndex < array.Length; nextIndex++)
            {
                if (array[nextIndex] >= 0)
                {
                    long cost = array[currentIndex] + Jump(array, jumpDistance, nextIndex, nextIndices, memoization);
                    if (cost < minimumCost)
                    {
                        minimumCost = cost;
                        nextIndices[currentIndex] = nextIndex;
                    }
                }
            }
            memoization[currentIndex] = minimumCost;
            return minimumCost;
        }
        public List<int> CheapestJumpNaive(int[] jumps, int maxJump)
        {
            int[] nextIndices = new int[jumps.Length];
            Array.Fill(nextIndices, -1);
            Jump(jumps, maxJump, 0, nextIndices);
            List<int> result = new List<int>();
            int index;
            for (index = 0; index < jumps.Length && nextIndices[index] > 0; index = nextIndices[index])
                result.Add(index + 1);
            if (index == jumps.Length - 1 && jumps[index] >= 0)
                result.Add(jumps.Length);
            else
                return new List<int>();
            return result;
        }

        private long Jump(int[] jumps, int maxJump, int currentIndex, int[] nextIndices)
        {
            if (currentIndex == jumps.Length - 1 && jumps[currentIndex] >= 0)
                return jumps[currentIndex];
            long minimumCost = int.MaxValue;
            for (int nextIndex = currentIndex + 1; nextIndex <= currentIndex + maxJump && nextIndex < jumps.Length; nextIndex++)
            {
                if (jumps[nextIndex] >= 0)
                {
                    long cost = jumps[currentIndex] + Jump(jumps, maxJump, nextIndex, nextIndices);
                    if (cost < minimumCost)
                    {
                        minimumCost = cost;
                        nextIndices[currentIndex] = nextIndex;
                    }
                }
            }
            return minimumCost;
        }

        /*
        256. Paint House
https://leetcode.com/problems/paint-house/description/

        */
        public int MinCostToPaintHouse(int[][] costs)
        {
            /*
Approach 1: Brute force
Complexity Analysis
•	Time complexity : O(2^n) or O(3^n).
Without writing code, we can get a good idea of the cost. We know that at the very least, we'd have to process every valid permutation. The number of valid permutations doubles with every house added. With 4 houses, there were 24 permutations. If we add another house, then all of our permutations for 4 houses could be extended with 2 different colors for the 5th house, giving 48 permutations. Because it doubles every time, this is O(n2).
It'd be even worse if we generated all permutations of 0, 1, and 2 and then pruned out the invalid ones. There are O(n3) such permutations in total.
•	Space complexity : Anywhere from O(n) to O(n⋅3n).
This would depend entirely on the implementation. If you generated all the permutations at the same time and put them in a massive list, then you'd be using O(n∗2n) or O(n∗3n) space. If you generated one, processed it, generated the next, processed it, etc, without keeping the long list, it'd require O(n) space.

            
            */

            /*
Approach 2: Brute force with a Recursive Tree
 Complexity Analysis
•	Time complexity : O(2^n).
While this approach is an improvement on the previous approach, it still requires exponential time. Think about the number of leaf nodes. Each permutation has its own leaf node. The number of internal nodes is the same as the number of leaf nodes too. Remember how there are 2n different permutations? Each effectively adds 2 nodes to the tree, so dropping the constant of 2 gives us O(2n).
This is better than the previous approach, which had an additional factor of n, giving O(n⋅2n). That extra factor of n has disappeared here because the permutations are now "sharing" their similar parts, unlike before. The idea of "sharing" similar parts can be taken much further for this particular problem, as we will see with the remaining approaches that knock the time complexity all the way down to O(n).
•	Space complexity : O(n).
This algorithm might initially appear to be O(1), because we are not allocating any new data structures. However, we need to take into account space usage on the run-time stack. The run-time stack was shown in the animation. Whenever we are processing the last house (house number n - 1), there are n stack frames on the stack. This space usages counts for complexity analysis (it's memory usage, like any other memory usage) and so the space complexity is O(n).           
            
            */
            int minCost = MinCostToPaintHouseNaiveReco(costs);

            /*
            
  Approach 3: Memoization
Complexity Analysis
•	Time complexity : O(n).
Analyzing memoization algorithms can be tricky at first, and requires understanding how recursion impacts the cost differently to loops. The key thing to notice is that the full function runs once for each possible set of parameters. There are 3 * n different possible sets of parameters, because there are n houses and 3 colors. Because the function body is O(1) (it's simply a conditional), this gives us a total of 3 * n. There can't be more than 3 * 2 * n searches into the memoization dictionary either. The tree showed this clearly—the nodes representing lookups had to be the child of a call where a full calculation was done. Because the constants are all dropped, this leaves O(n).
•	Space complexity : O(n).
Like the previous approach, the main space usage is on the stack. When we go down the first branch of function calls (see the tree visualization), we won't find any results in the dictionary. Therefore, every house will make a stack frame. Because there are n houses, this gives a worst case space usage of O(n). Note that this could be a problem in languages such as Python, where stack frames are large.

            */
            minCost = MinCostMemo(costs);
            /*
Approach 4: Dynamic Programming
 Complexity Analysis
•	Time Complexity : O(n).
Finding the minimum of two values and adding it to another value is an O(1) operation. We are doing these O(1) operations for 3⋅(n−1) cells in the grid. Expanding that out, we get 3⋅n−3. The constants don't matter in big-oh notation, so we drop them, leaving us with O(n).
A word of warning: This would not be correct if there were m colors. For this particular problem we were told there's only 3 colors. However, a logical follow-up question would be to make the code work for any number of colors. In that case, the time complexity would actually be O(n⋅m), because m is not a constant, whereas 3 is. If this confused you, I'd recommend reading up on big-oh notation.
•	Space Complexity : O(1)
We don't allocate any new data structures, and are only using a few local variables. All the work is done directly into the input array. Therefore, the algorithm is in-place, requiring constant extra space.
            
            */
            minCost = MinCostToPaintHouseDP(costs);
            /*
Approach 5: Dynamic Programming with Optimized Space Complexity            
Complexity Analysis
•	Time Complexity : O(n).
Same as previous approach.
•	Space Complexity : O(1)
We're "remembering" up to 6 calculations at a time (using 2 x length-3 arrays). Because this is actually a constant, the space complexity is still O(1).
Like the time complexity though, this analysis is dependent on there being a constant number of colors (i.e. 3). If the problem was changed to be m colors, then the space complexity would become O(m) as we'd need to keep track of a couple of length-m arrays.

            */
            minCost = MinCostToPaintHouseDPOptimal(costs);

            return minCost;

        }
        private int[][] costs;

        public int MinCostToPaintHouseNaiveReco(int[][] costs)
        {
            if (costs.Length == 0)
            {
                return 0;
            }
            this.costs = costs;
            return Math.Min(PaintCost(0, 0), Math.Min(PaintCost(0, 1), PaintCost(0, 2)));
        }

        private int PaintCost(int n, int color)
        {
            int totalCost = costs[n][color];
            if (n == costs.Length - 1)
            {
            }
            else if (color == 0)
            { // Red
                totalCost += Math.Min(PaintCost(n + 1, 1), PaintCost(n + 1, 2));
            }
            else if (color == 1)
            { // Green
                totalCost += Math.Min(PaintCost(n + 1, 0), PaintCost(n + 1, 2));
            }
            else
            { // Blue
                totalCost += Math.Min(PaintCost(n + 1, 0), PaintCost(n + 1, 1));
            }
            return totalCost;
        }
        private Dictionary<string, int> memoDict;

        public int MinCostMemo(int[][] costs)
        {
            if (costs.Length == 0)
            {
                return 0;
            }
            this.costs = costs;
            this.memoDict = new Dictionary<string, int>();
            return Math.Min(PaintCostMemo(0, 0), Math.Min(PaintCostMemo(0, 1), PaintCostMemo(0, 2)));
        }

        private int PaintCostMemo(int n, int color)
        {
            if (memoDict.ContainsKey(GetKey(n, color)))
            {
                return memoDict[GetKey(n, color)];
            }
            int totalCost = costs[n][color];
            if (n == costs.Length - 1)
            {
            }
            else if (color == 0)
            { // Red
                totalCost += Math.Min(PaintCostMemo(n + 1, 1), PaintCostMemo(n + 1, 2));
            }
            else if (color == 1)
            { // Green
                totalCost += Math.Min(PaintCostMemo(n + 1, 0), PaintCostMemo(n + 1, 2));
            }
            else
            { // Blue
                totalCost += Math.Min(PaintCostMemo(n + 1, 0), PaintCostMemo(n + 1, 1));
            }
            memoDict[GetKey(n, color)] = totalCost;

            return totalCost;
        }

        // Convert a house number and color into a simple string key for the memo.
        private string GetKey(int n, int color)
        {
            return n.ToString() + " " + color.ToString();
        }

        public int MinCostToPaintHouseDP(int[][] costs)
        {

            for (int n = costs.Length - 2; n >= 0; n--)
            {
                // Total cost of painting the nth house red.
                costs[n][0] += Math.Min(costs[n + 1][1], costs[n + 1][2]);
                // Total cost of painting the nth house green.
                costs[n][1] += Math.Min(costs[n + 1][0], costs[n + 1][2]);
                // Total cost of painting the nth house blue.
                costs[n][2] += Math.Min(costs[n + 1][0], costs[n + 1][1]);
            }

            if (costs.Length == 0) return 0;

            return Math.Min(Math.Min(costs[0][0], costs[0][1]), costs[0][2]);
        }
        public int MinCostToPaintHouseDPOptimal(int[][] costs)
        {
            if (costs.Length == 0) return 0;

            int[] previousRow = costs[costs.Length - 1];

            for (int houseIndex = costs.Length - 2; houseIndex >= 0; houseIndex--)
            {
                int[] currentRow = (int[])costs[houseIndex].Clone();
                // Total cost of painting the nth house red.
                currentRow[0] += Math.Min(previousRow[1], previousRow[2]);
                // Total cost of painting the nth house green.
                currentRow[1] += Math.Min(previousRow[0], previousRow[2]);
                // Total cost of painting the nth house blue.
                currentRow[2] += Math.Min(previousRow[0], previousRow[1]);
                previousRow = currentRow;
            }

            return Math.Min(Math.Min(previousRow[0], previousRow[1]), previousRow[2]);
        }

        /*
        265. Paint House II
        https://leetcode.com/problems/paint-house-ii/description/	

        */
        public int MinCostToPaintHouseII(int[][] costs)
        {
            /*
Approach 1: Memoization
Complexity Analysis
•	Time complexity : O(n⋅k^2).
Determining the total time complexity of a recursive memoization algorithm requires looking at how many calls are made to the paint function, and how much each call costs (remember that the memoization lookups are O(1)). The function is called once for each possible pair of house number and color. This gives n⋅k calls. Then, each call has a loop that loops over each of the k colors. Therefore, we have n⋅k⋅k=n⋅k2 which is O(n⋅k2).
The part outside of the recursive function is O(k) and therefore does not impact the overall complexity.
•	Space complexity : O(n⋅k).
There are 2 different places memory is being used that we need to consider.
Firstly, the memoization is storing the answers for each pair of house number and color. There are n⋅k of these, and so O(n⋅k) memory used.
Secondly, we need to consider the memory used on the run-time stack. In the worst case, there's a stack frame for each house number on the stack. This is a total of O(n).
The O(n) is insignficant to the O(n⋅k), so we're left with a total of O(n⋅k).
            
            */
            int minCost = MinCostToPaintHouseIIMemo(costs);

            /*
Approach 2: Dynamic Programming
Complexity Analysis
•	Time complexity : O(n⋅k^2).
We iterate over each of the n⋅k cells. For each of the cells, we're finding the minimum of the k values in the row above, excluding the one that is in the same column. This operation is O(k). Multiplying this out, we get O(n⋅k^2).
•	Space complexity : O(1) if done in-place, O(n⋅k) if input is copied.
We're not creating any new data structures in the code above, and so it has a space complexity of O(1). This is, however, overwriting the given input, which might not be ideal in some situations.
If we don't want to overwrite the input, we could instead create a copy of it first and then do the calculations in the copy. This will require an additional O(n⋅k) space.

            
            */
            minCost = MinCostToPaintHouseIIDP(costs);

            /*
Approach 3: Dynamic Programming with O(k) additional Space.
Complexity Analysis
•	Time complexity : O(n⋅k^2).
Same as above.
•	Space complexity : O(k).
The previous row and the current row are represented as k-length arrays.
This approach does not modify the input grid.	
            
            */

            minCost = MinCostToPaintHouseIIDP2(costs);
            /*
Approach 4: Dynamic programming with Optimized Time
Complexity Analysis
•	Time complexity : O(n⋅k).
The first loop that finds the minimums of the first row is O(k) because it looks at each of the k values in the first row exactly once. The second loop is O(n⋅k) because the outer loop loops n times, and the inner loop loops k times. O(n⋅k)+O(k)=O(n⋅k). We know it is impossible to ever do better here, because we cannot solve the problem without at least looking at each of the n⋅k cells once.
•	Space complexity : O(1).
Like approach 2, this approach also modifies the input instead of allocating its own space.
            
            */
            minCost = MinCostToPaintHouseIIDPOptimal(costs);

            /*
Approach 5: Dynamic programming with Optimized Time and Space
Complexity Analysis
•	Time complexity : O(n⋅k).
Same as the previous approach.
•	Space complexity : O(1).
The only additional working memory we're using is a constant number of single-value variables to keep track of the 2 minimums in the current and previous row, and to calculate the cost of the current cell. Because the memory usage is constant, we say it is O(1). Unlike the previous approach one though, this one does not overwrite the input.

            
            */
            minCost = MinCostToPaintHouseIIDPOptimal2(costs);

            return minCost;

        }

        private int n;
        private int k;

        public int MinCostToPaintHouseIIMemo(int[][] costs)
        {
            if (costs.Length == 0) return 0;
            this.k = costs[0].Length;
            this.n = costs.Length;
            this.costs = costs;
            this.memoDict = new Dictionary<string, int>();
            int minCost = int.MaxValue;
            for (int color = 0; color < k; color++)
            {
                minCost = Math.Min(minCost, MemoSolve(0, color));
            }
            return minCost;
        }

        private int MemoSolve(int houseNumber, int color)
        {

            // Base case: There are no more houses after this one.
            if (houseNumber == n - 1)
            {
                return costs[houseNumber][color];
            }

            // Memoization lookup case: Have we already solved this subproblem?
            if (memoDict.ContainsKey(GetKey(houseNumber, color)))
            {
                return memoDict[GetKey(houseNumber, color)];
            }

            // Recursive case: Determine the minimum cost for the remainder.
            int minRemainingCost = int.MaxValue;
            for (int nextColor = 0; nextColor < k; nextColor++)
            {
                if (color == nextColor) continue;
                int currentRemainingCost = MemoSolve(houseNumber + 1, nextColor);
                minRemainingCost = Math.Min(currentRemainingCost, minRemainingCost);
            }
            int totalCost = costs[houseNumber][color] + minRemainingCost;
            memoDict[GetKey(houseNumber, color)] = totalCost;
            return totalCost;
        }
        public int MinCostToPaintHouseIIDP(int[][] costs)
        {

            if (costs.Length == 0) return 0;
            int k = costs[0].Length;
            int n = costs.Length;

            for (int house = 1; house < n; house++)
            {
                for (int color = 0; color < k; color++)
                {
                    int mini = int.MaxValue;
                    for (int previousColor = 0; previousColor < k; previousColor++)
                    {
                        if (color == previousColor) continue;
                        mini = Math.Min(mini, costs[house - 1][previousColor]);
                    }
                    costs[house][color] += mini;
                }
            }

            // Find the minimum in the last row.
            int min = int.MaxValue;
            foreach (int c in costs[n - 1])
            {
                min = Math.Min(min, c);
            }
            return min;
        }

        public int MinCostToPaintHouseIIDP2(int[][] costs)
        {

            if (costs.Length == 0) return 0;
            int k = costs[0].Length;
            int n = costs.Length;
            int[] previousRow = costs[0];

            for (int house = 1; house < n; house++)
            {
                int[] currentRow = new int[k];
                for (int color = 0; color < k; color++)
                {
                    int minim = int.MaxValue;
                    for (int previousColor = 0; previousColor < k; previousColor++)
                    {
                        if (color == previousColor) continue;
                        minim = Math.Min(minim, previousRow[previousColor]);
                    }
                    currentRow[color] += costs[house][color] += minim;
                }
            }

            // Find the minimum in the last row.
            int min = int.MaxValue;
            foreach (int c in previousRow)
            {
                min = Math.Min(min, c);
            }
            return min;
        }
        public int MinCostToPaintHouseIIDPOptimal(int[][] costs)
        {

            if (costs.Length == 0) return 0;
            int k = costs[0].Length;
            int n = costs.Length;

            for (int house = 1; house < n; house++)
            {

                // Find the minimum and second minimum color in the PREVIOUS row.
                int minColor = -1; int secondMinColor = -1;
                for (int color = 0; color < k; color++)
                {
                    int cost = costs[house - 1][color];
                    if (minColor == -1 || cost < costs[house - 1][minColor])
                    {
                        secondMinColor = minColor;
                        minColor = color;
                    }
                    else if (secondMinColor == -1 || cost < costs[house - 1][secondMinColor])
                    {
                        secondMinColor = color;
                    }
                }

                // And now calculate the new costs for the current row.
                for (int color = 0; color < k; color++)
                {
                    if (color == minColor)
                    {
                        costs[house][color] += costs[house - 1][secondMinColor];
                    }
                    else
                    {
                        costs[house][color] += costs[house - 1][minColor];
                    }
                }
            }

            // Find the minimum in the last row.
            int min = int.MaxValue;
            foreach (int c in costs[n - 1])
            {
                min = Math.Min(min, c);
            }
            return min;
        }

        public int MinCostToPaintHouseIIDPOptimal2(int[][] costs)
        {

            if (costs.Length == 0) return 0;
            int k = costs[0].Length;
            int n = costs.Length;

            /* Firstly, we need to determine the 2 lowest costs of
              the first row. We also need to remember the color of the lowest. 
            */
            int prevMin = -1; int prevSecondMin = -1; int prevMinColor = -1;
            for (int color = 0; color < k; color++)
            {
                int cost = costs[0][color];
                if (prevMin == -1 || cost < prevMin)
                {
                    prevSecondMin = prevMin;
                    prevMinColor = color;
                    prevMin = cost;
                }
                else if (prevSecondMin == -1 || cost < prevSecondMin)
                {
                    prevSecondMin = cost;
                }
            }

            // And now, we need to work our way down, keeping track of the minimums.
            for (int house = 1; house < n; house++)
            {
                int min = -1; int secondMin = -1; int minColor = -1;
                for (int color = 0; color < k; color++)
                {
                    // Determine the cost for this cell (without writing it in).
                    int cost = costs[house][color];
                    if (color == prevMinColor)
                    {
                        cost += prevSecondMin;
                    }
                    else
                    {
                        cost += prevMin;
                    }
                    // Determine whether or not this current cost is also a minimum.
                    if (min == -1 || cost < min)
                    {
                        secondMin = min;
                        minColor = color;
                        min = cost;
                    }
                    else if (secondMin == -1 || cost < secondMin)
                    {
                        secondMin = cost;
                    }
                }
                // Transfer current mins to be previous mins.
                prevMin = min;
                prevSecondMin = secondMin;
                prevMinColor = minColor;
            }

            return prevMin;
        }

        /* 1473. Paint House III
        https://leetcode.com/problems/paint-house-iii/description/
         */
        public class MinCosToPaintHouseIIItSol
        {
            // Assign the size as per maximum value for different params
            private int?[][][] memo = new int?[100][][];
            // Maximum cost possible plus 1
            private const int MAX_COST = 1000001;

            public MinCosToPaintHouseIIItSol()
            {
                for (int i = 0; i < 100; i++)
                {
                    memo[i] = new int?[100][];
                    for (int j = 0; j < 100; j++)
                    {
                        memo[i][j] = new int?[21];
                    }
                }
            }
            /*
Approach 1: Top-Down Dynamic Programming
Complexity Analysis
Here, M is the number of houses, N is the number of colors and T is the number of target neighborhoods.
•	Time complexity: O(M⋅T⋅N^2)
Each state is defined by the values currIndex, neighborhoodCount, and prevHouseColor. Hence, there will be M⋅T⋅N possible states, and in the worst-case scenario, we must visit most of the states to solve the original problem. Each recursive call requires O(N) time as we might need to iterate over all the colors. Thus, the total time complexity is equal to O(M⋅T⋅N^2).
•	Space complexity: O(M⋅T⋅N)
The memoization results are stored in the table memo with size M⋅T⋅N. Also, stack space in the recursion is equal to the maximum number of active functions. The maximum number of active functions will be at most M i.e., one function call for every house. Hence, the space complexity is O(M⋅T⋅N).

*/
            public int TopDownDP(int[] houses, int[][] cost, int m, int n, int target)
            {
                int answer = FindMinCost(houses, cost, target, 0, 0, 0);
                // Return -1 if the answer is MAX_COST as it implies no answer possible
                return answer == MAX_COST ? -1 : answer;
            }
            private int FindMinCost(int[] houses, int[][] cost, int targetCount, int currIndex,
                                    int neighborhoodCount, int prevHouseColor)
            {
                if (currIndex == houses.Length)
                {
                    // If all houses are traversed, check if the neighbor count is as expected or not
                    return neighborhoodCount == targetCount ? 0 : MAX_COST;
                }

                if (neighborhoodCount > targetCount)
                {
                    // If the neighborhoods are more than the threshold, we can't have target neighborhoods
                    return MAX_COST;
                }

                // We have already calculated the answer so no need to go into recursion
                if (memo[currIndex][neighborhoodCount][prevHouseColor] != null)
                {
                    return memo[currIndex][neighborhoodCount][prevHouseColor].Value;
                }

                int minCost = MAX_COST;
                // If the house is already painted, update the values accordingly
                if (houses[currIndex] != 0)
                {
                    int newNeighborhoodCount = neighborhoodCount + (houses[currIndex] != prevHouseColor ? 1 : 0);
                    minCost =
                        FindMinCost(houses, cost, targetCount, currIndex + 1, newNeighborhoodCount, houses[currIndex]);
                }
                else
                {
                    int totalColors = cost[0].Length;

                    // If the house is not painted, try every possible color and store the minimum cost
                    for (int color = 1; color <= totalColors; color++)
                    {
                        int newNeighborhoodCount = neighborhoodCount + (color != prevHouseColor ? 1 : 0);
                        int currCost = cost[currIndex][color - 1]
                            + FindMinCost(houses, cost, targetCount, currIndex + 1, newNeighborhoodCount, color);
                        minCost = Math.Min(minCost, currCost);
                    }
                }

                // Return the minimum cost and also storing it for future reference (memoization)
                return (int)(memo[currIndex][neighborhoodCount][prevHouseColor] = minCost);
            }

            /*
            Approach 2: Bottom-Up Dynamic Programming
            Complexity Analysis
            Here, M is the number of houses, N is the number of colors and T is the number of target neighborhoods.
            •	Time complexity: O(M⋅T⋅N^2)
            Each state is defined by the values house, neighborhoods, and color. Hence, there will be M⋅T⋅N possible states, and in the worst-case scenario, we must visit most of the states to solve the original problem. Each state (subproblem) requires O(N) time as we iterate over all the colors for prevColor. Thus, the total time complexity is equal to O(M⋅T⋅N^2).
            •	Space complexity: O(M⋅T⋅N)
            The results are stored in the table memo with size M⋅T⋅N. Hence, the space complexity is equal to O(M⋅T⋅N).

            */

            public int BottomUpDP(int[] houses, int[][] cost, int m, int n, int target)
            {
                int[][][] memo = new int[m][][];
                for (int i = 0; i < m; i++)
                {
                    memo[i] = new int[target + 1][];
                    for (int j = 0; j <= target; j++)
                    {
                        memo[i][j] = new int[n];
                        Array.Fill(memo[i][j], MAX_COST);
                    }
                }

                // Initialize for house 0, neighborhoods will be 1
                for (int color = 1; color <= n; color++)
                {
                    if (houses[0] == color)
                    {
                        // If the house has same color, no cost
                        memo[0][1][color - 1] = 0;
                    }
                    else if (houses[0] == 0)
                    {
                        // If the house is not painted, assign the corresponding cost
                        memo[0][1][color - 1] = cost[0][color - 1];
                    }
                }

                for (int house = 1; house < m; house++)
                {
                    for (int neighborhoods = 1; neighborhoods <= Math.Min(target, house + 1); neighborhoods++)
                    {
                        for (int color = 1; color <= n; color++)
                        {
                            // If the house is already painted, and color is different
                            if (houses[house] != 0 && color != houses[house])
                            {
                                // Cannot be painted with different color
                                continue;
                            }

                            int currentCost = MAX_COST;
                            // Iterate over all the possible color for previous house
                            for (int previousColor = 1; previousColor <= n; previousColor++)
                            {
                                if (previousColor != color)
                                {
                                    // Decrement the neighborhood as adjacent houses has different color
                                    currentCost = Math.Min(currentCost, memo[house - 1][neighborhoods - 1][previousColor - 1]);
                                }
                                else
                                {
                                    // Previous house has the same color, no change in neighborhood count
                                    currentCost = Math.Min(currentCost, memo[house - 1][neighborhoods][color - 1]);
                                }
                            }

                            // If the house is already painted, cost to paint is 0
                            int costToPaint = houses[house] != 0 ? 0 : cost[house][color - 1];
                            memo[house][neighborhoods][color - 1] = currentCost + costToPaint;
                        }
                    }
                }

                int minimumCost = MAX_COST;
                // Find the minimum cost with m houses and target neighborhoods
                // By comparing cost for different color for the last house
                for (int color = 1; color <= n; color++)
                {
                    minimumCost = Math.Min(minimumCost, memo[m - 1][target][color - 1]);
                }

                // Return -1 if the answer is MAX_COST as it implies no answer possible
                return minimumCost == MAX_COST ? -1 : minimumCost;
            }
            /*
            Approach 3: Bottom-Up Dynamic Programming (Space Optimized)
Complexity Analysis
Here, M is the number of houses, N is the number of colors and T is the number of target neighborhoods.
•	Time complexity: O(M⋅T⋅N^2)
We are iterating over the houses from 1 to M and for each house, store the results in the table memo by iterating over each neighbor and color. Therefore, we have T⋅N states for each house, and each such state will take O(N) operations to iterate over the prevColor options. Hence the total time complexity is O(M⋅T⋅N^2).
•	Space complexity: O(T⋅N)
The results are stored in the arrays memo and prevMemo, each with a size of T⋅N. Hence, the space complexity equals O(T⋅N).

            */
            public int BottomUpDPSpaceOptimal(int[] houses, int[][] cost, int m, int n, int target)
            {
                int[][] prevMemo = new int[target + 1][];
                for (int i = 0; i <= target; i++)
                {
                    prevMemo[i] = new int[n];
                    for (int j = 0; j < n; j++)
                    {
                        prevMemo[i][j] = MAX_COST;
                    }
                }

                // Initialize for house 0, neighborhood will be 1
                for (int color = 1; color <= n; color++)
                {
                    if (houses[0] == color)
                    {
                        // If the house has same color, no cost
                        prevMemo[1][color - 1] = 0;
                    }
                    else if (houses[0] == 0)
                    {
                        // If the house is not painted, assign the corresponding cost
                        prevMemo[1][color - 1] = cost[0][color - 1];
                    }
                }

                for (int house = 1; house < m; house++)
                {
                    int[][] memo = new int[target + 1][];
                    for (int i = 0; i <= target; i++)
                    {
                        memo[i] = new int[n];
                        for (int j = 0; j < n; j++)
                        {
                            memo[i][j] = MAX_COST;
                        }
                    }

                    for (int neighborhoods = 1; neighborhoods <= Math.Min(target, house + 1); neighborhoods++)
                    {
                        for (int color = 1; color <= n; color++)
                        {
                            // If the house is already painted, and color is different
                            if (houses[house] != 0 && color != houses[house])
                            {
                                // Cannot be painted with different color
                                continue;
                            }

                            int currentCost = MAX_COST;
                            // Iterate over all the possible color for previous house
                            for (int prevColor = 1; prevColor <= n; prevColor++)
                            {
                                if (prevColor != color)
                                {
                                    // Decrement the neighborhood as adjacent houses has different color
                                    currentCost = Math.Min(currentCost, prevMemo[neighborhoods - 1][prevColor - 1]);
                                }
                                else
                                {
                                    // Previous house has the same color, no change in neighborhood count
                                    currentCost = Math.Min(currentCost, prevMemo[neighborhoods][color - 1]);
                                }
                            }

                            // If the house is already painted cost to paint is 0
                            int costToPaint = houses[house] != 0 ? 0 : cost[house][color - 1];
                            memo[neighborhoods][color - 1] = currentCost + costToPaint;
                        }
                    }
                    // Update the table to have the current house results
                    prevMemo = memo;
                }

                int minCost = MAX_COST;
                // Find the minimum cost with m houses and target neighborhoods
                // By comparing cost for different color for the last house
                for (int color = 1; color <= n; color++)
                {
                    minCost = Math.Min(minCost, prevMemo[target][color - 1]);
                }

                // Return -1 if the answer is MAX_COST as it implies no answer possible
                return minCost == MAX_COST ? -1 : minCost;
            }

        }

        /*
        276. Paint Fence
        https://leetcode.com/problems/paint-fence/description/
        */

        public int NumWaysToPaintFence(int n, int k)
        {
            /*            
Approach 1: Top-Down Dynamic Programming (Recursion + Memoization) 
Complexity Analysis
•	Time complexity: O(n)
totalWays gets called with each index from n to 3. Because of our memoization, each call will only take O(1) time.
•	Space complexity: O(n)
The extra space used by this algorithm is the recursion call stack. For example, totalWays(50) will call totalWays(49), which calls totalWays(48) etc., all the way down until the base cases at totalWays(1) and totalWays(2). In addition, our hash map memo will be of size n at the end, since we populate it with every index from n to 3.


            */
            int numWays = NumWaysToPaintFenceTDDPRecMemo(n, k);
            /*
Approach 2: Bottom-Up Dynamic Programming (Tabulation)
Complexity Analysis
•	Time complexity: O(n)
We only iterate from 3 to n once, where each iteration requires O(1) time.
•	Space complexity: O(n)
We need to use an array totalWays, where totalWays.length scales linearly with n.


            */
            numWays = NumWaysToPaintFenceBUDP(n, k);
            /*
Approach 3: Bottom-Up, Constant Space
Complexity Analysis
•	Time complexity: O(n).
We only iterate from 3 to n once, each time doing O(1) work.
•	Space complexity: O(1)
The only extra space we use are a few integer variables, which are independent of input size
          

            */
            numWays = NumWaysToPaintFenceBUDPOptimal(n, k);

            return numWays;
        }
        private Dictionary<int, int> memoDict1 = new Dictionary<int, int>();

        private int TotalWays(int i, int k)
        {
            if (i == 1) return k;
            if (i == 2) return k * k;

            // Check if we have already calculated totalWays(i)
            if (memoDict1.ContainsKey(i))
            {
                return memoDict1[i];
            }

            // Use the recurrence relation to calculate totalWays(i)
            memoDict1[i] = (k - 1) * (TotalWays(i - 1, k) + TotalWays(i - 2, k));
            return memoDict1[i];
        }

        public int NumWaysToPaintFenceTDDPRecMemo(int n, int k)
        {
            return TotalWays(n, k);
        }

        public int NumWaysToPaintFenceBUDP(int n, int k)
        {
            // Base cases for the problem to avoid index out of bound issues
            if (n == 1) return k;
            if (n == 2) return k * k;

            int[] totalWaysArr = new int[n + 1];
            totalWaysArr[1] = k;
            totalWaysArr[2] = k * k;

            for (int i = 3; i <= n; i++)
            {
                totalWaysArr[i] = (k - 1) * (totalWaysArr[i - 1] + totalWaysArr[i - 2]);

            }
            return totalWaysArr[n];
        }
        public int NumWaysToPaintFenceBUDPOptimal(int n, int k)
        {
            if (n == 1) return k;

            int twoPostsBack = k;
            int onePostBack = k * k;

            for (int i = 3; i <= n; i++)
            {
                int curr = (k - 1) * (onePostBack + twoPostsBack);
                twoPostsBack = onePostBack;
                onePostBack = curr;
            }

            return onePostBack;
        }


        /*

278. First Bad Version
https://leetcode.com/problems/first-bad-version/description/

        */

        public int FirstBadVersion(int n)
        {
            /*
Approach #1 (Linear Scan) [Time Limit Exceeded]
Complexity analysis
•	Time complexity : O(n).
Assume that isBadVersion(version) takes constant time to check if a version is bad. It takes at most n−1 checks, therefore the overall time complexity is O(n).
•	Space complexity : O(1).


            */
            int firstBadVersion = FirstBadVersionNaive(n);
            /*
  Approach #2 Binary Search (BS) [Accepted]          
   Complexity analysis
•	Time complexity : O(logn).
The search space is halved each time, so the time complexity is O(logn).
•	Space complexity : O(1).
         
            */

            firstBadVersion = FirstBadVersionBS(n);

            return firstBadVersion;

        }

        public int FirstBadVersionNaive(int n)
        {
            for (int i = 1; i < n; i++)
            {
                if (IsBadVersion(i))
                {
                    return i;
                }
            }
            return n;
        }

        public int FirstBadVersionBS(int n)
        {
            int left = 1;
            int right = n;
            while (left < right)
            {
                int mid = left + (right - left) / 2;
                if (IsBadVersion(mid))
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

        bool IsBadVersion(int version)
        {
            return false; //dummy implementation; It cam be true also 
        }
        /*
        765. Couples Holding Hands
        https://leetcode.com/problems/couples-holding-hands/description/

        */
        public int MinSwapsCouples(int[] row)
        {
            /*
Approach #1: Backtracking [Time Limit Exceeded]
 Complexity Analysis
•	Time Complexity: O(N∗2^N), where N is the number of couples, as for each couch we check up to two complete possibilities. The N factor is from searching for jx and jy; this factor can be removed with a more efficient algorithm that keeps track of where pairs[j][k] is x as we swap elements through pairs.
•	Space Complexity: O(N).
           
            */
            int minSwapsCouples = MinSwapsCouplesBacktrack(row);

            /*
Approach #2: Cycle Finding [Accepted] (CF)
Complexity Analysis
•	Time Complexity: O(N), where N is the number of couples.
•	Space Complexity: O(N), the size of adj and associated data structures.

            */
            minSwapsCouples = MinSwapsCouplesCF(row);
            /*
  Approach #3: Greedy [Accepted]          
   Complexity Analysis
•	Time Complexity: O(N^2), where N is the number of couples.
•	Space Complexity: O(1) additional complexity: the swaps are in place
         
            */
            minSwapsCouples = MinSwapsCouplesGreedy(row);

            return minSwapsCouples;

        }

        int N;
        int[][] pairs;

        public int MinSwapsCouplesBacktrack(int[] row)
        {
            N = row.Length / 2;
            pairs = new int[N][];
            for (int i = 0; i < N; ++i)
            {
                pairs[i][0] = row[2 * i] / 2;
                pairs[i][1] = row[2 * i + 1] / 2;
            }

            return solve(0);
        }

        public void swap(int a, int b, int c, int d)
        {
            int t = pairs[a][b];
            pairs[a][b] = pairs[c][d];
            pairs[c][d] = t;
        }

        public int solve(int i)
        {
            if (i == N) return 0;
            int x = pairs[i][0], y = pairs[i][1];
            if (x == y) return solve(i + 1);

            int jx = 0, kx = 0, jy = 0, ky = 0; // Always gets set later
            for (int j = i + 1; j < N; ++j)
            {
                for (int k = 0; k <= 1; ++k)
                {
                    if (pairs[j][k] == x) { jx = j; kx = k; }
                    if (pairs[j][k] == y) { jy = j; ky = k; }
                }
            }

            swap(i, 1, jx, kx);
            int ans1 = 1 + solve(i + 1);
            swap(i, 1, jx, kx);

            swap(i, 0, jy, ky);
            int ans2 = 1 + solve(i + 1);
            swap(i, 0, jy, ky);

            return Math.Min(ans1, ans2);
        }


        public int MinSwapsCouplesCF(int[] row)
        {
            int N = row.Length / 2;
            //couples[x] = {i, j} means that
            //couple #x is at couches i and j (1 indexed)
            int[][] couples = new int[N][];

            for (int i = 0; i < row.Length; ++i)
                Add(couples[row[i] / 2], i / 2 + 1);

            //adj[x] = {i, j} means that
            //x-th couch connected to couches i, j (all 1 indexed) by couples
            int[][] adj = new int[N + 1][];
            foreach (int[] couple in couples)
            {
                Add(adj[couple[0]], couple[1]);
                Add(adj[couple[1]], couple[0]);
            }

            // The answer will be N minus the number of cycles in adj.
            int ans = N;
            // For each couch (1 indexed)
            for (int r = 1; r <= N; ++r)
            {
                // If this couch has no people needing to be paired, continue
                if (adj[r][0] == 0 && adj[r][1] == 0)
                    continue;

                // Otherwise, there is a cycle starting at couch r.
                // We will use two pointers x, y with y faster than x by one turn.
                ans--;
                int x = r, y = Pop(adj[r]);
                // When y reaches the start 'r', we've reached the end of the cycle.
                while (y != r)
                {
                    // We are at some couch with edges going to 'x' and 'new'.
                    // We remove the previous edge, since we came from x.
                    Rem(adj[y], x);

                    // We update x, y appropriately: y becomes new and x becomes y.
                    x = y;
                    y = Pop(adj[y]);
                }
            }
            return ans;
        }

        // Replace the next zero element with x.
        public void Add(int[] pair, int x)
        {
            pair[pair[0] == 0 ? 0 : 1] = x;
        }

        // Remove x from pair, replacing it with zero.
        public void Rem(int[] pair, int x)
        {
            pair[pair[0] == x ? 0 : 1] = 0;
        }

        // Remove the next non-zero element from pair, replacing it with zero.
        public int Pop(int[] pair)
        {
            int x = pair[0];
            if (x != 0)
            {
                pair[0] = 0;
            }
            else
            {
                x = pair[1];
                pair[1] = 0;
            }
            return x;
        }


        public int MinSwapsCouplesGreedy(int[] row)
        {
            int ans = 0;
            for (int i = 0; i < row.Length; i += 2)
            {
                int x = row[i];
                if (row[i + 1] == (x ^ 1)) continue;
                ans++;
                for (int j = i + 1; j < row.Length; ++j)
                {
                    if (row[j] == (x ^ 1))
                    {
                        row[j] = row[i + 1];
                        row[i + 1] = x ^ 1;
                        break;
                    }
                }
            }
            return ans;
        }

        /*
        55. Jump Game
        https://leetcode.com/problems/jump-game/

        */
        public bool CanJump(int[] nums)
        {

            /*

Approach 1: Backtracking
Complexity Analysis
•	Time complexity : O(2^n). There are 2^n (upper bound) ways of jumping from the first position to the last, where n is the length of array nums. For a complete proof, please refer to Appendix A.
•	Space complexity : O(n). Recursion requires additional memory for the stack frames

            
            */
            bool canJump = CanJumpBacktrack(nums);
            /*
 Approach 2: Dynamic Programming Top-down (DPTD)           
 Complexity Analysis
•	Time complexity : O(n^2).
For every element in the array, say i, we are looking at the next nums[i] elements to its right aiming to find a GOOD index. nums[i] can be at most n, where n is the length of array nums.
•	Space complexity : O(2^n)=O(n).
First n originates from recursion. Second n comes from the usage of the memo table.
           
            */
            canJump = CanJumpDPTD(nums);
            /*
Approach 3: Dynamic Programming Bottom-up (DPBU)
Complexity Analysis
•	Time complexity : O(n^2).
For every element in the array, say i, we are looking at the next nums[i] elements to its right aiming to find a GOOD index. nums[i] can be at most n, where n is the length of array nums.
•	Space complexity : O(n).
This comes from the usage of the memo table.

            
            */
            canJump = CanJumpDPBU(nums);
            /*
 Approach 4: Greedy
 Complexity Analysis
•	Time complexity : O(n).
We are doing a single pass through the nums array, hence n steps, where n is the length of array nums.
•	Space complexity : O(1).
We are not using any extra memory
           
            */
            canJump = CanJumpGreedy(nums);

            return canJump;

        }
        public bool CanJumpFromPosition(int position, int[] nums)
        {
            if (position == nums.Length - 1)
            {
                return true;
            }

            int furthestJump = Math.Min(position + nums[position], nums.Length - 1);
            for (int nextPosition = position + 1; nextPosition <= furthestJump;
                 nextPosition++)
            {
                if (CanJumpFromPosition(nextPosition, nums))
                {
                    return true;
                }
            }

            return false;
        }

        public bool CanJumpBacktrack(int[] nums)
        {
            return CanJumpFromPosition(0, nums);
        }

        public enum Index { GOOD, BAD, UNKNOWN }

        public bool CanJumpFromPosition(int position, int[] nums, Index[] memo)
        {
            if (memo[position] != Index.UNKNOWN)
            {
                return memo[position] == Index.GOOD ? true : false;
            }

            int furthestJump = Math.Min(position + nums[position], nums.Length - 1);
            for (int nextPosition = position + 1; nextPosition <= furthestJump;
                 nextPosition++)
            {
                if (CanJumpFromPosition(nextPosition, nums))
                {
                    memo[position] = Index.GOOD;
                    return true;
                }
            }

            memo[position] = Index.BAD;
            return false;
        }

        public bool CanJumpDPTD(int[] nums)
        {
            Index[] memo = new Index[nums.Length];
            for (int i = 0; i < memo.Length; i++)
            {
                memo[i] = Index.UNKNOWN;
            }

            memo[memo.Length - 1] = Index.GOOD;
            return CanJumpFromPosition(0, nums, memo);
        }
        public bool CanJumpDPBU(int[] nums)
        {
            Index[] memo = new Index[nums.Length];
            for (int i = 0; i < memo.Length; i++)
            {
                memo[i] = Index.UNKNOWN;
            }

            memo[memo.Length - 1] = Index.GOOD;
            for (int i = nums.Length - 2; i >= 0; i--)
            {
                int furthestJump = Math.Min(i + nums[i], nums.Length - 1);
                for (int j = i + 1; j <= furthestJump; j++)
                {
                    if (memo[j] == Index.GOOD)
                    {
                        memo[i] = Index.GOOD;
                        break;
                    }
                }
            }

            return memo[0] == Index.GOOD;
        }

        public bool CanJumpGreedy(int[] nums)
        {
            int lastPos = nums.Length - 1;
            for (int i = nums.Length - 1; i >= 0; i--)
            {
                if (i + nums[i] >= lastPos)
                {
                    lastPos = i;
                }
            }

            return lastPos == 0;
        }

        /*
        45. Jump Game II
        https://leetcode.com/problems/jump-game-ii/description/

        Approach 1: Greedy
        Complexity Analysis
        Let n be the length of the input array nums.
        •	Time complexity: O(n)
        We iterate over nums and stop at the second last element. In each step of the iteration, we make some calculations that take constant time. Therefore, the overall time complexity is O(n).
        •	Space complexity: O(1)
        In the iteration, we only need to update three variables, curEnd, curFar and answer, they only take constant space.
        */
        public int Jump(int[] nums)
        {
            // The starting range of the first jump is [0, 0]
            int answer = 0, n = nums.Length;
            int curEnd = 0, curFar = 0;
            for (int i = 0; i < n - 1; ++i)
            {
                // Update the farthest reachable index of this jump.
                curFar = Math.Max(curFar, i + nums[i]);
                // If we finish the starting range of this jump,
                // Move on to the starting range of the next jump.
                if (i == curEnd)
                {
                    answer++;
                    curEnd = curFar;
                }
            }

            return answer;

        }
        /*
        1306. Jump Game III
https://leetcode.com/problems/jump-game-iii/description/
        */
        public bool CanReach(int[] arr, int start)
        {
            /*
            
Approach 1: Breadth-First Search
Complexity Analysis
Assume N is the length of arr.
•	Time complexity: O(N) since we will visit every index at most once.
•	Space complexity : O(N) since it needs q to store next index. In fact, q would keep at most two levels of nodes. Since we got two children for each node, the traversal of this solution is a binary tree. The maximum number of nodes within a single level for a binary tree would be 2N, so the maximum length of q is
O(N/2+N/2)=O(N).

            */
            bool canReach = CanReachBFS(arr, start);
            /*
Approach 2: Depth-First Search
 Complexity Analysis
Assume N is the length of arr.
•	Time complexity: O(N), since we will visit every index only once.
•	Space complexity: O(N) since it needs at most O(N) stacks for recursions.
           
            */
            canReach = CanReachDFS(arr, start);

            return canReach;

        }
        public bool CanReachBFS(int[] arr, int start)
        {
            int n = arr.Length;

            Queue<int> q = new Queue<int>();
            q.Enqueue(start);

            while (q.Count > 0)
            {
                int node = q.Dequeue();
                // check if reach zero
                if (arr[node] == 0)
                {
                    return true;
                }
                if (arr[node] < 0)
                {
                    continue;
                }

                // check available next steps
                if (node + arr[node] < n)
                {
                    q.Enqueue(node + arr[node]);
                }
                if (node - arr[node] >= 0)
                {
                    q.Enqueue(node - arr[node]);
                }
                // mark as visited
                arr[node] = -arr[node];
            }
            return false;
        }
        public bool CanReachDFS(int[] arr, int start)
        {
            if (start >= 0 && start < arr.Length && arr[start] >= 0)
            {
                if (arr[start] == 0)
                {
                    return true;
                }
                arr[start] = -arr[start];
                return CanReachDFS(arr, start + arr[start]) || CanReachDFS(arr, start - arr[start]);
            }
            return false;
        }

        /*
        1345. Jump Game IV
        https://leetcode.com/problems/jump-game-iv/description/

        */
        public int MinJumps(int[] arr)
        {
            /*
            Approach 1: Breadth-First Search
            Complexity Analysis
            Assume N is the length of arr.
            •	Time complexity: O(N) since we will visit every node at most once.
            •	Space complexity: O(N) since it needs curs and nex to store nodes.


            */
            int minJumps = MinJumpsBFS(arr);
            /*
            Approach 2: Bidirectional BFS
            Complexity Analysis
            Assume N is the length of arr.
            •	Time complexity: O(N) since we will visit every node at most once, but usually faster than approach 1.
            •	Space complexity: O(N) since it needs curs, other and nex to store nodes.

            */
            minJumps = MinJumpsBBFS(arr);

            return minJumps;
        }

        public int MinJumpsBFS(int[] arr)
        {
            int n = arr.Length;
            if (n <= 1)
            {
                return 0;
            }

            Dictionary<int, List<int>> graph = new Dictionary<int, List<int>>();
            for (int i = 0; i < n; i++)
            {
                graph[arr[i]] = new List<int>();
            }

            List<int> curs = new List<int>(); // store current layer
            curs.Add(0);
            HashSet<int> visited = new HashSet<int>();
            int step = 0;

            // when current layer exists
            while (curs.Count > 0)
            {
                List<int> nex = new List<int>();

                // iterate the layer
                foreach (int node in curs)
                {
                    // check if reached end
                    if (node == n - 1)
                    {
                        return step;
                    }

                    // check same value
                    foreach (int child in graph[arr[node]])
                    {
                        if (!visited.Contains(child))
                        {
                            visited.Add(child);
                            nex.Add(child);
                        }
                    }

                    // clear the list to prevent redundant search
                    graph[arr[node]].Clear();

                    // check neighbors
                    if (node + 1 < n && !visited.Contains(node + 1))
                    {
                        visited.Add(node + 1);
                        nex.Add(node + 1);
                    }
                    if (node - 1 >= 0 && !visited.Contains(node - 1))
                    {
                        visited.Add(node - 1);
                        nex.Add(node - 1);
                    }
                }

                curs = nex;
                step++;
            }

            return -1;
        }


        public int MinJumpsBBFS(int[] arr)
        {
            int n = arr.Length;
            if (n <= 1)
            {
                return 0;
            }

            Dictionary<int, List<int>> graph = new Dictionary<int, List<int>>();
            for (int i = 0; i < n; i++)
            {
                graph[arr[i]] = new List<int>
                {
                    i
                };
            }

            HashSet<int> curs = new HashSet<int>(); // store layers from start
            curs.Add(0);
            HashSet<int> visited = new HashSet<int>();
            visited.Add(0);
            visited.Add(n - 1);
            int step = 0;

            HashSet<int> other = new HashSet<int>(); //store layers from end
            other.Add(n - 1);

            // when current layer exists
            while (curs.Count > 0)
            {
                // search from the side with fewer nodes
                if (curs.Count > other.Count)
                {
                    HashSet<int> tmp = curs;
                    curs = other;
                    other = tmp;
                }
                HashSet<int> nex = new HashSet<int>();

                // iterate the layer
                foreach (int node in curs)
                {

                    // check same value
                    foreach (int child in graph[arr[node]])
                    {
                        if (other.Contains(child))
                            return step + 1;

                        if (!visited.Contains(child))
                        {
                            visited.Add(child);
                            nex.Add(child);
                        }
                    }

                    // clear the list to prevent redundant search
                    graph[arr[node]].Clear();

                    // check neighbors.
                    if (other.Contains(node + 1) || other.Contains(node - 1))
                        return step + 1;

                    if (node + 1 < n && !visited.Contains(node + 1))
                    {
                        visited.Add(node + 1);
                        nex.Add(node + 1);
                    }
                    if (node - 1 >= 0 && !visited.Contains(node - 1))
                    {
                        visited.Add(node - 1);
                        nex.Add(node - 1);
                    }
                }

                curs = nex;
                step++;
            }

            return -1;
        }

        /*
        1340. Jump Game V
        https://leetcode.com/problems/jump-game-v/description/

        Approach: DFS with Memo

        Complexity Analysis
        Time: O(nd)
        Memory: O(n) to memoize jumps for every index.

        */
        public int MaxJumps(int[] arr, int d)
        {
            int[] memo = new int[arr.Length];//default 0
            for (int i = 0; i < arr.Length; i++)
            {
                MaxJumpsDFSRec(i, memo, d, arr);
            }
            return memo.Max();
        }
        private int MaxJumpsDFSRec(int i, int[] dp, int d, int[] arr)
        {
            if (dp[i] != 0) return dp[i];//already calculated, return it
            dp[i] = 1;
            for (int j = i + 1; j < arr.Length && j <= i + d; j++)
            {
                if (arr[i] > arr[j]) dp[i] = Math.Max(dp[i], 1 + MaxJumpsDFSRec(j, dp, d, arr));
                else break;
            }
            for (int j = i - 1; 0 <= j && j >= i - d; j--)
            {
                if (arr[i] > arr[j]) dp[i] = Math.Max(dp[i], 1 + MaxJumpsDFSRec(j, dp, d, arr));
                else break;
            }
            return dp[i];
        }

        /*
        1696. Jump Game VI	
        https://leetcode.com/problems/jump-game-vi/description/	


        */
        public int MaxResult(int[] nums, int k)
        {

            /*
            Approach 1: Dynamic Programming + Deque (DPD)
            Complexity Analysis
            Let N be the length of nums.
            •	Time Complexity: O(N), since we need to iterate nums, and push and pop each element into the deque at most once.
            •	Space Complexity: O(N), since we need O(N) space to store our dp array and O(k) to store dq.


            */
            int maxResult = MaxResultDPD(nums, k);
            /*
            Approach 2: Dynamic Programming + Priority Queue (DPPQ)
            Complexity Analysis
            Let N be the length of nums.
            •	Time Complexity: O(NlogN), since we need to iterate nums, and push and pop each element into the deque at most once, and for each push and pop, it costs O(logN) in the worst case.
            •	Space Complexity: O(N), since we need O(N) space to store our dp array and O(N) to store priority_queue.


            */
            maxResult = MaxResultDPPQ(nums, k);

            /*
            Approach 3: Segment Tree (ST)
            Complexity Analysis
            Let N be the length of nums.
            •	Time Complexity: O(NlogN), since we need to iterate nums, and for each element we need to perform the query and update once, which costs O(logN) in the worst case.
            •	Space Complexity: O(N), since we need O(N) space to store our segment tree.


            */
            maxResult = MaxResultST(nums, k);
            /*
            Approach 4: Dynamic Programming + Deque (Compressed) (DPDC)
            Complexity Analysis
            Let N be the length of nums.
            •	Time Complexity: O(N), since we need to iterate nums, and push and pop each element into the deque at most once.
            •	Space Complexity: O(k), since we need O(k) to store dq


            */
            maxResult = MaxResultDPDC(nums, k);

            /*
            Approach 5: Dynamic Programming + Priority Queue (Compressed) (DPPQC)
            Complexity Analysis
            Let N be the length of nums.
            •	Time Complexity: O(NlogN), since we need to iterate nums, and push and pop each element into the deque at most once, and for each push and pop, it costs O(logN) in the worst case.
            •	Space Complexity: O(N), since we need O(N) to store priority_queue.

            */
            maxResult = MaxResultDPPQC(nums, k);

            return maxResult;

        }

        public int MaxResultDPD(int[] nums, int k)
        {
            int n = nums.Length;
            int[] score = new int[n];
            score[0] = nums[0];
            LinkedList<int> dq = new LinkedList<int>();
            dq.AddLast(0);
            for (int i = 1; i < n; i++)
            {
                // pop the old index
                while (dq.First != null && dq.First.Value < i - k)
                {
                    dq.RemoveFirst();
                }
                score[i] = score[dq.First.Value] + nums[i];
                // pop the smaller value
                while (dq.Last != null && score[i] >= score[dq.Last.Value])
                {
                    dq.RemoveLast();
                }
                dq.AddLast(i);
            }
            return score[n - 1];
        }

        public int MaxResultDPPQ(int[] nums, int k)
        {
            int n = nums.Length;
            int[] score = new int[n];
            score[0] = nums[0];
            PriorityQueue<int[], int> pq = new PriorityQueue<int[], int>();
            //PriorityQueue<int[], int> priorityQueue = new PriorityQueue<int[], int>((a, b) => b[0] - a[0]);

            pq.Enqueue(new int[] { nums[0], 0 }, -nums[0]); //Using MinHeap as MaxHeap in disguise using negator

            for (int i = 1; i < n; i++)
            {
                // pop the old index
                while (pq.Peek()[1] < i - k)
                {
                    pq.Dequeue();
                }
                score[i] = nums[i] + score[pq.Peek()[1]];
                pq.Enqueue(new int[] { score[i], i }, -score[i]);
            }
            return score[n - 1];
        }

        public int MaxResultST(int[] nums, int k)
        {
            int n = nums.Length;
            int[] segmentTree = new int[2 * n];
            Update(0, nums[0], segmentTree, n);
            for (int i = 1; i < n; i++)
            {
                int maxQueryResult = Query(Math.Max(0, i - k), i, segmentTree, n);
                Update(i, maxQueryResult + nums[i], segmentTree, n);
            }
            return segmentTree[2 * n - 1];
        }

        // implement Segment Tree
        private void Update(int index, int value, int[] segmentTree, int n)
        {
            index += n;
            segmentTree[index] = value;
            for (index >>= 1; index > 0; index >>= 1)
            {
                segmentTree[index] = Math.Max(segmentTree[index << 1], segmentTree[(index << 1) + 1]);
            }
        }

        private int Query(int left, int right, int[] segmentTree, int n)
        {
            int result = int.MinValue;
            for (left += n, right += n; left < right; left >>= 1, right >>= 1)
            {
                if ((left & 1) == 1)
                {
                    result = Math.Max(result, segmentTree[left++]);
                }
                if ((right & 1) == 1)
                {
                    result = Math.Max(result, segmentTree[--right]);
                }
            }
            return result;
        }
        public int MaxResultDPDC(int[] nums, int k)
        {
            int n = nums.Length;
            int currentScore = nums[0];
            LinkedList<int[]> dq = new LinkedList<int[]>();
            dq.AddLast(new int[] { 0, currentScore });
            for (int index = 1; index < n; index++)
            {
                // pop the old index
                while (dq.First() != null && dq.First()[0] < index - k)
                {
                    dq.RemoveFirst();
                }
                currentScore = dq.First()[1] + nums[index];
                // pop the smaller value
                while (dq.Last() != null && currentScore >= dq.Last()[1])
                {
                    dq.RemoveLast();
                }
                dq.AddLast(new int[] { index, currentScore });
            }
            return currentScore;
        }

        public int MaxResultDPPQC(int[] nums, int k)
        {
            int n = nums.Length;
            int currentScore = nums[0];
            PriorityQueue<int[], int> priorityQueue = new PriorityQueue<int[], int>();
            priorityQueue.Enqueue(new int[] { nums[0], 0 }, -0); //Using MinHeap as MaxHeap in disguise using negator
            for (int index = 1; index < n; index++)
            {
                // pop the old index
                while (priorityQueue.Peek()[1] < index - k)
                {
                    priorityQueue.Dequeue();
                }
                currentScore = nums[index] + priorityQueue.Peek()[0];
                priorityQueue.Enqueue(new int[] { currentScore, index }, -index);
            }
            return currentScore;
        }

        /*
        1871. Jump Game VII
        https://leetcode.com/problems/jump-game-vii/description/

        Approach: One Pass DP
        Complexity
        Time O(n)
        Space O(n)

        */
        public bool CanReach(string s, int minJ, int maxJ)
        {
            int n = s.Length, pre = 0;
            bool[] dp = new bool[n];
            dp[0] = true;
            for (int i = 1; i < n; ++i)
            {
                if (i >= minJ && dp[i - minJ])
                    pre++;
                if (i > maxJ && dp[i - maxJ - 1])
                    pre--;
                dp[i] = pre > 0 && s[i] == '0';
            }
            return dp[n - 1];
        }

        /*
        2297. Jump Game VIII
        https://leetcode.com/problems/jump-game-viii/description/	

        Approach: Monotonic Stack + Dynamic Programming,
        Complextiy:
        Time : O(n)*/

        public long MinCost(int[] nums, int[] costs)
        {
            int n = nums.Length;

            // 1. build graph with monotonic stack
            // each element will be pushed to and popped out of stack at most twice, this with will be O(n)
            List<int>[] graph = new List<int>[n];
            for (int i = 0; i < n; i++)
            {
                graph[i] = new List<int>();
            }

            Stack<int> stack = new Stack<int>();
            stack.Push(0);
            for (int i = 1; i < n; i++)
            {
                while (stack.Count > 0 && nums[stack.Peek()] > nums[i])
                {
                    graph[stack.Pop()].Add(i);
                }
                stack.Push(i);
            }

            stack = new Stack<int>();
            stack.Push(0);
            for (int i = 1; i < n; i++)
            {
                while (stack.Count > 0 && nums[stack.Peek()] <= nums[i])
                {
                    graph[stack.Pop()].Add(i);
                }
                stack.Push(i);
            }


            // 2. DP to find min cost to reach each index
            // one node can have two neighbors, next smaller and next bigger
            // so this part is O(n)
            long[] dp = new long[n];
            Array.Fill(dp, long.MaxValue);
            dp[0] = 0;
            for (int cur = 0; cur < n; cur++)
            {
                foreach (int next in graph[cur])
                {
                    dp[next] = Math.Min(dp[next], dp[cur] + costs[next]);
                }
            }
            return dp[n - 1];
        }

        /*
        1730. Shortest Path to Get Food
        https://leetcode.com/problems/shortest-path-to-get-food/description/

        Complexity:
        TIME: O(M*N)
        Space: O(M*N)
        */
        public int GetFood(char[][] grid)
        {
            Queue<(int x, int y)> queue = new Queue<(int x, int y)>();

            for (int row = 0; row < grid.Length; row++)
            {
                for (int column = 0; column < grid[0].Length; column++)
                {
                    if (grid[row][column] == '*')
                    {
                        queue.Enqueue((row, column));
                        grid[row][column] = '-';
                    }
                }
            }

            int distance = 0;
            int[] directionX = { -1, 1, 0, 0 };
            int[] directionY = { 0, 0, -1, 1 };

            while (queue.Count > 0)
            {
                int count = queue.Count;
                distance++;

                for (int i = 0; i < count; i++)
                {
                    var current = queue.Dequeue();

                    for (int direction = 0; direction < 4; direction++)
                    {
                        int nextRow = current.x + directionX[direction];
                        int nextColumn = current.y + directionY[direction];

                        if (nextRow < 0 || nextRow >= grid.Length || nextColumn < 0 || nextColumn >= grid[0].Length || grid[nextRow][nextColumn] == 'X' || grid[nextRow][nextColumn] == '-')
                            continue;

                        if (grid[nextRow][nextColumn] == '#')
                            return distance;

                        grid[nextRow][nextColumn] = '-';
                        queue.Enqueue((nextRow, nextColumn));
                    }
                }
            }

            return -1;
        }

        /*
        48. Rotate Image	
        https://leetcode.com/problems/rotate-image/description/	

        */
        public void RotateImage(int[][] matrix)
        {
            /*
            Approach 1: Rotate Groups of Four Cells (RGFC)
            Complexity Analysis
Let M be the number of cells in the matrix.
•	Time complexity: O(M), as each cell is getting read once and written once.
•	Space complexity: O(1) because we do not use any other additional data structures.

            */
            RotateImageRGFC(matrix);
            /*
            Approach 2: Reverse on the Diagonal and then Reverse Left to Right (RDRLR)
            Complexity Analysis
    Let M be the number of cells in the grid.
    •	Time complexity: O(M). We perform two steps; transposing the matrix, and then reversing each row. Transposing the matrix has a cost of O(M) because we're moving the value of each cell once. Reversing each row also has a cost of O(M), because again we're moving the value of each cell once.
    •	Space complexity: O(1) because we do not use any other additional data structures.

            */

            RotateImageRDRLR(matrix);
        }


        public void RotateImageRGFC(int[][] matrix)
        {
            if (matrix == null || matrix.Length == 0 || matrix.Any(row => row.Length != matrix.Length))
            {
                throw new ArgumentException("Input must be a square matrix.");
            }

            int n = matrix.Length;
            for (int ringIndex = 0; ringIndex < (n + 1) / 2; ringIndex++)
            {
                for (int elementIndex = 0; elementIndex < n / 2; elementIndex++)
                {
                    RotateRingClockwise(matrix, ringIndex, elementIndex, n);
                }
            }
        }

        //helper method that rotates a single "ring" of elements in a square matrix by 90 degrees clockwise
        //n the context of matrix rotation, a "ring" refers to a set of elements that form a rectangular loop around the matrix. The outermost ring consists of the top row, right column, bottom row, and left column of the matrix. The next inner ring consists of the elements just inside the outermost ring, and so on
        private void RotateRingClockwise(int[][] matrix, int ringIndex, int elementIndex, int n)
        {
            // Store the current element in a temp variable
            int temp = matrix[n - 1 - elementIndex][ringIndex];

            // Rotate the elements in the current "ring" by 90 degrees clockwise
            matrix[n - 1 - elementIndex][ringIndex] = matrix[n - 1 - ringIndex][n - elementIndex - 1];
            matrix[n - 1 - ringIndex][n - elementIndex - 1] = matrix[elementIndex][n - 1 - ringIndex];
            matrix[elementIndex][n - 1 - ringIndex] = matrix[ringIndex][elementIndex];
            matrix[ringIndex][elementIndex] = temp;

        }

        public void RotateImageRDRLR(int[][] matrix)
        {
            int n = matrix.Length;
            Transpose(matrix, n);
            Reflect(matrix, n);
        }

        private void Transpose(int[][] matrix, int n)
        {
            for (int i = 0; i < n; i++)
            {
                for (int j = i + 1; j < n; j++)
                {
                    int temp = matrix[j][i];
                    matrix[j][i] = matrix[i][j];
                    matrix[i][j] = temp;
                }
            }
        }

        private void Reflect(int[][] matrix, int n)
        {
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n / 2; j++)
                {
                    int temp = matrix[i][j];
                    matrix[i][j] = matrix[i][n - j - 1];
                    matrix[i][n - j - 1] = temp;
                }
            }
        }


        /*
        242. Valid Anagram
        https://leetcode.com/problems/valid-anagram/description/
        */
        public bool IsAnagram(string s, string t)
        {
            /*
       Approach 1: Sorting     
        Complexity Analysis
•	Time complexity: O(nlogn).
Assume that n is the length of s, sorting costs O(nlogn) and comparing two strings costs O(n). Sorting time dominates and the overall time complexity is O(nlogn).
•	Space complexity: O(1).
Space depends on the sorting implementation which, usually, costs O(1) auxiliary space if heapsort is used. Note that in Java, toCharArray() makes a copy of the string so it costs O(n) extra space, but we ignore this for complexity analysis because:
o	It is a language dependent detail.
o	It depends on how the function is designed. For example, the function parameter types can be changed to char[].
    
            */
            bool isAnagram = IsAnagramSort(s, t);

            /*
  Approach 2: Frequency Counter  (FC)      
Complexity Analysis
•	Time complexity: O(n).
Time complexity is O(n) because accessing the counter table is a constant time operation.
•	Space complexity: O(1).
Although we do use extra space, the space complexity is O(1) because the table's size stays constant no matter how large n is.

            */
            isAnagram = IsAnagramFC(s, t);

            return isAnagram;

        }
        public bool IsAnagramSort(String s, String t)
        {
            if (s.Length != t.Length)
            {
                return false;
            }
            char[] str1 = s.ToCharArray();
            char[] str2 = t.ToCharArray();
            Array.Sort(str1);
            Array.Sort(str2);
            return new string(str1) == new string(str2);
        }
        public bool IsAnagramFC(String s, String t)
        {
            if (s.Length != t.Length)
            {
                return false;
            }
            int[] table = new int[26];
            for (int i = 0; i < s.Length; i++)
            {
                table[s[i] - 'a']++;
            }
            for (int i = 0; i < t.Length; i++)
            {
                table[t[i] - 'a']--;
                if (table[t[i] - 'a'] < 0)
                {
                    return false;
                }
            }
            return true;
        }

        /*
        438. Find All Anagrams in a String
        https://leetcode.com/problems/find-all-anagrams-in-a-string/description/

        */
        public IList<int> FindAnagrams(string input, string pattern)
        {
            /*
            Approach 1: Sliding Window with HashMap (SWHM)
Complexity Analysis
Let Ns and Np be the length of s and p respectively. Let K be the maximum possible number of distinct characters. In this problem, K equals 26 because s and p consist of lowercase English letters.
•	Time complexity: O(Ns)
We perform one pass along each string when Ns≥Np which costs O(Ns+Np) time. Since we only perform this step when Ns≥Np the time complexity simplifies to O(Ns).
•	Space complexity: O(K)
pCount and sCount will contain at most K elements each. Since K is fixed at 26 for this problem, this can be considered as O(1) space.

            */
            var anagramsFound = FindAnagramsSWHM(input, pattern);
            /*
        Approach 2: Sliding Window with Array (SWA)
 Complexity Analysis
Let Ns and Np be the length of s and p respectively. Let K be the maximum possible number of distinct characters. In this problem, K equals 26 because s and p consist of lowercase English letters.
•	Time complexity: O(Ns)
We perform one pass along each string when Ns≥Np which costs O(Ns+Np) time. Since we only perform this step when Ns≥Np the time complexity simplifies to O(Ns).
•	Space complexity: O(K)
pCount and sCount contain K elements each. Since K is fixed at 26 for this problem, this can be considered as O(1) space.
       
            */
            anagramsFound = FindAnagramsSWA(input, pattern);

            return anagramsFound;

        }
        public IList<int> FindAnagramsSWHM(string s, string p)
        {
            int stringLength = s.Length, patternLength = p.Length;
            if (stringLength < patternLength) return new List<int>();

            Dictionary<char, int> patternCount = new Dictionary<char, int>();
            Dictionary<char, int> stringCount = new Dictionary<char, int>();

            // Build a reference hashmap using string p
            foreach (char character in p)
            {
                if (patternCount.ContainsKey(character))
                {
                    patternCount[character]++;
                }
                else
                {
                    patternCount[character] = 1;
                }
            }

            List<int> output = new List<int>();

            // The sliding window on the string s
            for (int i = 0; i < stringLength; ++i)
            {
                // Add one more letter 
                // on the right side of the window
                char character = s[i];
                if (stringCount.ContainsKey(character))
                {
                    stringCount[character]++;
                }
                else
                {
                    stringCount[character] = 1;
                }

                // Remove one letter 
                // from the left side of the window
                if (i >= patternLength)
                {
                    character = s[i - patternLength];
                    if (stringCount[character] == 1)
                    {
                        stringCount.Remove(character);
                    }
                    else
                    {
                        stringCount[character]--;
                    }
                }

                // Compare hashmap in the sliding window
                // with the reference hashmap
                if (AreDictionariesEqual(patternCount, stringCount))
                {
                    output.Add(i - patternLength + 1);
                }
            }
            return output;
        }

        private bool AreDictionariesEqual(Dictionary<char, int> dict1, Dictionary<char, int> dict2)
        {
            if (dict1.Count != dict2.Count) return false;
            foreach (var kvp in dict1)
            {
                if (!dict2.ContainsKey(kvp.Key) || dict2[kvp.Key] != kvp.Value)
                {
                    return false;
                }
            }
            return true;
        }
        public List<int> FindAnagramsSWA(string inputString, string pattern)
        {
            int inputStringLength = inputString.Length, patternLength = pattern.Length;
            if (inputStringLength < patternLength) return new List<int>();

            int[] patternCount = new int[26];
            int[] inputStringCount = new int[26];
            // build reference array using string pattern
            foreach (char character in pattern)
            {
                patternCount[(int)(character - 'a')]++;
            }

            List<int> resultIndices = new List<int>();
            // sliding window on the string inputString
            for (int index = 0; index < inputStringLength; ++index)
            {
                // add one more letter 
                // on the right side of the window
                inputStringCount[(int)(inputString[index] - 'a')]++;
                // remove one letter 
                // from the left side of the window
                if (index >= patternLength)
                {
                    inputStringCount[(int)(inputString[index - patternLength] - 'a')]--;
                }
                // compare array in the sliding window
                // with the reference array
                if (AreArraysEqual(patternCount, inputStringCount))
                {
                    resultIndices.Add(index - patternLength + 1);
                }
            }
            return resultIndices;
        }

        private bool AreArraysEqual(int[] array1, int[] array2)
        {
            for (int i = 0; i < array1.Length; i++)
            {
                if (array1[i] != array2[i])
                {
                    return false;
                }
            }
            return true;
        }
        /*
        49. Group Anagrams
        https://leetcode.com/problems/group-anagrams/editorial/

        */
        public IList<IList<string>> GroupAnagrams(string[] strs)
        {
            /*
Approach 1: Categorize by Sorted String (CSS)
Complexity Analysis
•	Time Complexity: O(NKlogK), where N is the length of strs, and K is the maximum length of a string in strs. The outer loop has complexity O(N) as we iterate through each string. Then, we sort each string in O(KlogK) time.
•	Space Complexity: O(NK), the total information content stored in ans.
            
            */
            var groupedAnagrams = GroupAnagramsCSS(strs);

            /* 
Approach 2: Categorize by Count (CC)
Complexity Analysis
•	Time Complexity: O(NK), where N is the length of strs, and K is the maximum length of a string in strs. Counting each string is linear in the size of the string, and we count every string.
•	Space Complexity: O(NK), the total information content stored in ans.           
            
            */
            groupedAnagrams = GroupAnagramsCC(strs);

            return groupedAnagrams;

        }
        public IList<IList<string>> GroupAnagramsCSS(string[] strs)
        {
            var dict = new Dictionary<string, List<string>>();
            foreach (var s in strs)
            {
                var ca = s.ToCharArray();
                Array.Sort(ca);
                var key = new String(ca);
                if (!dict.ContainsKey(key))
                    dict[key] = new List<string>();
                dict[key].Add(s);
            }

            return new List<IList<string>>(dict.Values);
        }
        public IList<IList<string>> GroupAnagramsCC(string[] strs)
        {
            if (strs.Length == 0)
                return new List<IList<string>>();  // an empty string
            Dictionary<string, List<string>> ans =
                new Dictionary<string, List<string>>();
            int[] count = new int[26];
            foreach (string s in strs)
            {
                for (int i = 0; i < 26; ++i)
                {
                    count[i] = 0;
                }

                // Increase the count as per char
                foreach (char c in s) count[c - 'a']++;
                StringBuilder sb = new StringBuilder("");
                for (int i = 0; i < 26; i++)
                {
                    sb.Append('#');
                    sb.Append(count[i]);
                }

                string key = sb.ToString();
                if (!ans.ContainsKey(key))
                    ans[key] = new List<string>();
                ans[key].Add(s);
            }

            return new List<IList<string>>(ans.Values);
        }

        /*
        436. Find Right Interval
        https://leetcode.com/problems/find-right-interval/description/

        */
        public int[] FindRightInterval(int[][] intervals)
        {

            /*
Approach 1: Brute Force
Complexity Analysis
•	Time complexity : O(n2). The complete set of n intervals is scanned for every(n) interval chosen.
•	Space complexity : O(n). res array of size n is used.

            */
            int[] rightInterval = FindRightIntervalNaive(intervals);
            /*
Approach 2: Using Sorting + Scanning (SS)
 Complexity Analysis
•	Time complexity : O(n^2).
o	Sorting takes O(nlogn) time.
o	For the first interval we need to search among n−1 elements.
o	For the second interval, the search is done among n−2 elements and so on leading to a total of: (n−1)+(n−2)+...+1=(n.(n−1))/ 2=O(n^2) calculations.
•	Space complexity : O(n). res array of size n is used. A hashmap hash of size n is used.
           
            */
            rightInterval = FindRightIntervalSS(intervals);
            /*
Approach 3: Using Sorting + Binary Search (SBS)
Complexity Analysis
•	Time complexity : O(nlogn). Sorting takes O(nlogn) time. Binary search takes O(logn) time for each of the n intervals.
•	Space complexity : O(n). res array of size n is used. A hashmap hash of size O(n) is used.
            
            */
            rightInterval = FindRightIntervalSBS(intervals);
            /*

Approach 4: Using TreeMap /SortedDictionary in C# (SD)
   Complexity Analysis
•	Time complexity : O(N⋅logN). Inserting an element into TreeMap takes O(logN) time. N such insertions are done. The search in TreeMap using ceilingEntry also takes O(logN) time. N such searches are done.
•	Space complexity : O(N). res array of size n is used. TreeMap starts of size O(N) is used.
         
            */
            rightInterval = FindRightIntervalSD(intervals);
            /*
 Approach 5: Using Two Arrays without Binary Search (TA)          
 Complexity Analysis
•	Time complexity : O(N⋅logN). Sorting takes O(N⋅logN) time. A total of O(N) time is spent on searching for the appropriate intervals, since the endIntervals and intervals array is scanned only once.
•	Space complexity : O(N). endIntervals, intervals and res array of size N are used. A hashmap hash of size O(N) is used.
           
            */
            rightInterval = FindRightIntervalTA(intervals);


            return rightInterval;

        }
        public int[] FindRightIntervalNaive(int[][] intervals)
        {
            int[] res = new int[intervals.Length];
            for (int i = 0; i < intervals.Length; i++)
            {
                int min = int.MaxValue;
                int minindex = -1;
                for (int j = 0; j < intervals.Length; j++)
                {
                    if (intervals[j][0] >= intervals[i][1] && intervals[j][0] < min)
                    {
                        min = intervals[j][0];
                        minindex = j;
                    }
                }
                res[i] = minindex;
            }
            return res;
        }

        public int[] FindRightIntervalSS(int[][] intervals)
        {
            int[] res = new int[intervals.Length];
            Dictionary<int[], int> hash = new Dictionary<int[], int>();

            for (int i = 0; i < intervals.Length; i++)
            {
                hash.Add(intervals[i], i);
            }
            Array.Sort(intervals, (a, b) => a[0].CompareTo(b[0]));
            for (int i = 0; i < intervals.Length; i++)
            {
                int min = int.MaxValue;
                int minindex = -1;
                for (int j = i; j < intervals.Length; j++)
                {
                    if (intervals[j][0] >= intervals[i][1] && intervals[j][0] < min)
                    {
                        min = intervals[j][0];
                        minindex = hash[intervals[j]];
                    }
                }
                res[hash[intervals[i]]] = minindex;
            }
            return res;
        }

        public int[] BinarySearch(int[][] intervals, int target, int start, int end)
        {
            if (start >= end)
            {
                if (intervals[start][0] >= target)
                {
                    return intervals[start];
                }
                return null;
            }
            int mid = (start + end) / 2;
            if (intervals[mid][0] < target)
            {
                return BinarySearch(intervals, target, mid + 1, end);
            }
            else
            {
                return BinarySearch(intervals, target, start, mid);
            }
        }

        public int[] FindRightIntervalSBS(int[][] intervals)
        {
            int[] resultArray = new int[intervals.Length];
            Dictionary<int[], int> intervalIndexMap = new Dictionary<int[], int>();
            for (int i = 0; i < intervals.Length; i++)
            {
                intervalIndexMap[intervals[i]] = i;
            }
            Array.Sort(intervals, (a, b) => a[0].CompareTo(b[0]));
            for (int i = 0; i < intervals.Length; i++)
            {
                int[] interval = BinarySearch(intervals, intervals[i][1], 0, intervals.Length - 1);
                resultArray[intervalIndexMap[intervals[i]]] = interval == null ? -1 : intervalIndexMap[interval];
            }
            return resultArray;
        }
        public int[] FindRightIntervalSD(int[][] intervals)
        {
            SortedDictionary<int, int> startTimes = new SortedDictionary<int, int>();
            int[] result = new int[intervals.Length];
            for (int index = 0; index < intervals.Length; index++)
            {
                startTimes[intervals[index][0]] = index;
            }
            for (int index = 0; index < intervals.Length; index++)
            {
                KeyValuePair<int, int>? position = startTimes.FirstOrDefault(x => x.Key >= intervals[index][1]);
                result[index] = position.HasValue ? position.Value.Value : -1;
            }
            return result;
        }
        public int[] FindRightIntervalTA(int[][] intervals)
        {
            int[][] endIntervals = (int[][])intervals.Clone();
            Dictionary<int[], int> indexMap = new Dictionary<int[], int>();
            for (int i = 0; i < intervals.Length; i++)
            {
                indexMap[intervals[i]] = i;
            }
            Array.Sort(intervals, (a, b) => a[0].CompareTo(b[0]));
            Array.Sort(endIntervals, (a, b) => a[1].CompareTo(b[1]));
            int j = 0;
            int[] result = new int[intervals.Length];
            for (int i = 0; i < endIntervals.Length; i++)
            {
                while (j < intervals.Length && intervals[j][0] < endIntervals[i][1])
                {
                    j++;
                }
                result[indexMap[endIntervals[i]]] = j == intervals.Length ? -1 : indexMap[intervals[j]];
            }
            return result;
        }

        /*
    71. Simplify Path
https://leetcode.com/problems/simplify-path/description/
  Complexity Analysis
•	Time Complexity: O(N) if there are N characters in the original path. First, we spend O(N) trying to split the input path into components and then we process each component one by one which is again an O(N) operation. We can get rid of the splitting part and just string together the characters and form directory names etc. However, that would be too complicated and not worth depicting in the implementation. The main idea of this algorithm is to use a stack. How you decide to process the input string is a personal choice.
•	Space Complexity: O(N). Actually, it's 2N because we have the array that contains the split components and then we have the stack.
  
        */
        public string SimplifyUnixFilePath(string path)
        {
            // Initialize a stack
            Stack<string> stack = new Stack<string>();
            string[] components = path.Split('/');
            // Split the input string on "/" as the delimiter
            // and process each portion one by one
            foreach (string directory in components)
            {
                // A no-op for a "." or an empty string
                if (directory.Equals(".") || directory.Length == 0)
                {
                    continue;
                }
                else if (directory.Equals(".."))
                {
                    // If the current component is a "..", then
                    // we pop an entry from the stack if it's non-empty
                    if (stack.Any())
                    {
                        stack.Pop();
                    }
                }
                else
                {
                    // Finally, a legitimate directory name, so we add it
                    // to our stack
                    stack.Push(directory);
                }
            }

            // Stich together all the directory names together
            StringBuilder result = new StringBuilder();
            foreach (string dir in stack.Reverse())
            {
                result.Append("/");
                result.Append(dir);
            }

            return result.Length > 0 ? result.ToString() : "/";
        }

        /*
        72. Edit Distance
        https://leetcode.com/problems/edit-distance/description/

        */
        public class MinEditDistanceSol
        {
            /*          
        Approach 1: Recursion

Complexity Analysis
Let K be the length of string word1 and N be the length of string word2. Let M=max(K,N).
•	Time Complexity: O(3^M)
o	The time complexity is exponential. For every pair of word1 and word2, if the characters do not match, we recursively explore three possibilities. In the worst case, if none of the characters match, we will end up exploring O(3^M) possibilities.
•	Space Complexity: O(M)
o	The recursion uses an internal call stack equal to the depth of the recursion tree. The recursive process will terminate when either word1 or word2 is empty.

            */
            public int MinEditDistanceNaiveRec(string word1, string word2)
            {
                return MinEditDistance(word1, word2, word1.Length, word2.Length);
            }

            private int MinEditDistance(string word1, string word2, int m, int n)
            {
                // If any string is empty, return the length of the other string
                if (m == 0)
                    return n;
                if (n == 0)
                    return m;

                // If the last characters are the same, no edit is needed
                if (word1[m - 1] == word2[n - 1])
                    return MinEditDistance(word1, word2, m - 1, n - 1);

                // Find the minimum of three possible operations
                return 1 + Math.Min(
                    MinEditDistance(word1, word2, m, n - 1), // Insert
                    Math.Min(
                        MinEditDistance(word1, word2, m - 1, n), // Remove
                        MinEditDistance(word1, word2, m - 1, n - 1) // Replace
                    )
                );
            }

            /*
            Approach 2: Memoization: Top-Down Dynamic Programming (DBPTD)
            Complexity Analysis
    Let M be the length of string word1 and N be the length of string word2.
    •	Time Complexity: O(M⋅N)
    o	As the memoization approach uses the cache, for every combination of word1 and word2 the result is computed only once.
    •	Space Complexity: O(M⋅N)
    o	The space is for the additional 2-dimensional array memo of size (M⋅N).

            */
            public int MinEditDistanceDPTD(string word1, string word2)
            {
                int?[][] memo = new int?[word1.Length + 1][];
                for (int i = 0; i <= word1.Length; i++)
                    memo[i] = new int?[word2.Length + 1];
                return MinDistanceRecur(word1, word2, word1.Length, word2.Length);

                int MinDistanceRecur(string word1, string word2, int word1Index,
                                     int word2Index)
                {
                    if (word1Index == 0)
                        return word2Index;
                    if (word2Index == 0)
                        return word1Index;
                    if (memo[word1Index][word2Index] != null)
                        return memo[word1Index][word2Index].Value;
                    int minEditDistance = 0;
                    if (word1[word1Index - 1] == word2[word2Index - 1])
                        minEditDistance = MinDistanceRecur(word1, word2, word1Index - 1,
                                                           word2Index - 1);
                    else
                    {
                        int insertOperation =
                            MinDistanceRecur(word1, word2, word1Index, word2Index - 1);
                        int deleteOperation =
                            MinDistanceRecur(word1, word2, word1Index - 1, word2Index);
                        int replaceOperation = MinDistanceRecur(
                            word1, word2, word1Index - 1, word2Index - 1);
                        minEditDistance =
                            Math.Min(insertOperation,
                                     Math.Min(deleteOperation, replaceOperation)) +
                            1;
                    }

                    memo[word1Index][word2Index] = minEditDistance;
                    return minEditDistance;
                }
            }

            /*
Approach 3: Bottom-Up Dynamic Programming (DPBU): Tabulation
Complexity Analysis
Let M be the length of string word1 and N be the length of string word2.
•	Time Complexity: O(M⋅N)
o	In the nested for loop, the outer loop iterates M times, and the inner loop iterates N times.
Thus, the time complexity is O(M⋅N).
•	Space Complexity: O(M⋅N)
o	The space is for the additional 2-dimensional array dp of size (M⋅N).


            */
            public int MinEditDistanceDPBU(string word1, string word2)
            {
                int word1Length = word1.Length;
                int word2Length = word2.Length;
                if (word1Length == 0)
                {
                    return word2Length;
                }

                if (word2Length == 0)
                {
                    return word1Length;
                }

                int[,] dp = new int[word1Length + 1, word2Length + 1];
                for (int word1Index = 1; word1Index <= word1Length; word1Index++)
                {
                    dp[word1Index, 0] = word1Index;
                }

                for (int word2Index = 1; word2Index <= word2Length; word2Index++)
                {
                    dp[0, word2Index] = word2Index;
                }

                for (int word1Index = 1; word1Index <= word1Length; word1Index++)
                {
                    for (int word2Index = 1; word2Index <= word2Length; word2Index++)
                    {
                        if (word2[word2Index - 1] == word1[word1Index - 1])
                        {
                            dp[word1Index, word2Index] =
                                dp[word1Index - 1, word2Index - 1];
                        }
                        else
                        {
                            dp[word1Index, word2Index] =
                                Math.Min(dp[word1Index - 1, word2Index],
                                         Math.Min(dp[word1Index, word2Index - 1],
                                                  dp[word1Index - 1, word2Index - 1])) +
                                1;
                        }
                    }
                }

                return dp[word1Length, word2Length];
            }


        }


        /* 161. One Edit Distance
        https://leetcode.com/problems/one-edit-distance/description/
         */
        public class IsOneEditDistanceSol
        {
            /*             
            Approach 1: One pass algorithm
            Complexity Analysis
•	Time complexity: O(N) in the worst case when string lengths are close enough abs(ns - nt) <= 1, where N is a number of characters in the longest string. O(1) in the best case when abs(ns - nt) > 1.
•	Space complexity: O(N) because strings are immutable in Python and Java and create substring costs O(N) space.
             */

            public bool IsOneEditDistance(string s, string t)
            {
                int ns = s.Length;
                int nt = t.Length;

                // Ensure that s is shorter than t.
                if (ns > nt)
                    return IsOneEditDistance(t, s);

                // The strings are NOT one edit away from distance  
                // if the length diff is more than 1.
                if (nt - ns > 1)
                    return false;

                for (int i = 0; i < ns; i++)
                    if (s[i] != t[i])
                        // If strings have the same length
                        if (ns == nt)
                            return s.Substring(i + 1) == t.Substring(i + 1);
                        // If strings have different lengths
                        else
                            return s.Substring(i) == t.Substring(i + 1);

                // If there are no diffs in ns distance
                // The strings are one edit away only if
                // t has one more character. 
                return ns + 1 == nt;
            }
        }


        /*
        Approach: Greedy

        Complexity Analysis
        Here, M is the number of strings in the bank and N is the average length of the strings.
        •	Time complexity: O(M∗N)
        We have to iterate over each character once to find the number of safety devices in each row and hence the time complexity is equal to O(M∗N).
        •	Space complexity: O(1)
        We only need three variables prev, ans and count and hence the space complexity is constant.

        */
        public int NumberOfBeams(string[] bank)
        {
            int prev = 0, ans = 0;

            foreach (string s in bank)
            {
                int count = 0;
                foreach (char c in s)
                {
                    if (c == '1')
                    {
                        count++;
                    }
                }
                if (count != 0)
                {
                    ans += (prev * count);
                    prev = count;
                }
            }

            return ans;

        }

        /*

468. Validate IP Address
https://leetcode.com/problems/validate-ip-address/description/

*/
        public class ValidIPAddressSol
        {
            /*Approach 1: Regex
            Complexity Analysis
•	Time complexity: O(1) because the patterns to match have
constant length.
•	Space complexity: O(1).

            */

            private static string chunkIPv4 = "([0-9]|[1-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5])";
            private static Regex patternIPv4 = new Regex("^(" + chunkIPv4 + @"\\.){3}" + chunkIPv4 + "$");

            private static string chunkIPv6 = "([0-9a-fA-F]{1,4})";
            private static Regex patternIPv6 = new Regex("^(" + chunkIPv6 + @"\:){7}" + chunkIPv6 + "$");

            public string ValidIPAddress(string IP)
            {
                if (patternIPv4.IsMatch(IP)) return "IPv4";
                return (patternIPv6.IsMatch(IP)) ? "IPv6" : "Neither";
            }


            /*
            Approach 2: Divide and Conquer
            Complexity Analysis
•	Time complexity: O(N) because to count number of dots requires to
parse the entire input string.
•	Space complexity: O(1).


            */
            public string ValidateIPv4(String IP)
            {
                string[] nums = IP.Split("\\.", -1);
                foreach (string x in nums)
                {
                    // Validate integer in range (0, 255):
                    // 1. length of chunk is between 1 and 3
                    if (x.Length == 0 || x.Length > 3) return "Neither";
                    // 2. no extra leading zeros
                    if (x[0] == '0' && x.Length != 1) return "Neither";
                    // 3. only digits are allowed
                    foreach (char ch in x.ToCharArray())
                    {
                        if (!char.IsAsciiDigit(ch)) return "Neither";
                    }
                    // 4. less than 255
                    if (int.Parse(x) > 255) return "Neither";
                }
                return "IPv4";
            }

            public String ValidateIPv6(String IP)
            {
                string[] nums = IP.Split(":", -1);
                string hexdigits = "0123456789abcdefABCDEF";
                foreach (string x in nums)
                {
                    // Validate hexadecimal in range (0, 2**16):
                    // 1. at least one and not more than 4 hexdigits in one chunk
                    if (x.Length == 0 || x.Length > 4) return "Neither";
                    // 2. only hexdigits are allowed: 0-9, a-f, A-F
                    foreach (char ch in x.ToCharArray())
                    {
                        if (hexdigits.IndexOf(ch) == -1) return "Neither";
                    }
                }
                return "IPv6";
            }

            public String validIPAddress(String IP)
            {
                if (IP.ToCharArray().Select(ch => ch == '.').Count() == 3)
                {
                    return ValidateIPv4(IP);
                }
                else if (IP.ToCharArray().Select(ch => ch == ':').Count() == 7)
                {
                    return ValidateIPv6(IP);
                }
                else return "Neither";
            }

        }



        /*
            93. Restore IP Addresses
            https://leetcode.com/problems/restore-ip-addresses/description/
            https://www.algoexpert.io/questions/valid-ip-addresses

            */
        public class RestoreIPAddressesSol
        {


            /*
            Approach 1: Backtracking
            Complexity Analysis
        Let's assume we need to separate the input string into N integers, each integer is at most M digits.
        •	Time complexity: O(M^N⋅N).
        There are at most MN−1 possibilities, and for each possibility checking whether all parts are valid takes O(M⋅N) time, so the final time complexity is O(M^(N−1))⋅O(M⋅N) = O(M^N⋅N).
        For this question, M = 3, N = 4, so the time complexity is O(1).
        •	Space complexity: O(M⋅N).
        For each possibility, we save (N - 1) numbers (the number of digits before each dot) which takes O(N) space. And we need temporary space to save a solution before putting it into the answer list. The length of each solution string is M⋅N+M−1 = O(M⋅N), so the total space complexity is O(M⋅N) if we don't take the output space into consideration.
        For this question, M = 3, N = 4, so the space complexity is O(1).

            */
            private bool Valid(string s, int start, int length)
            {
                return length == 1 ||
                       (s[start] != '0' &&
                        (length < 3 || int.Parse(s.Substring(start, length)) <= 255));
            }

            private void Helper(string s, int startIndex, List<int> dots,
                                List<string> ans)
            {
                var remainingLength = s.Length - startIndex;
                var remainingNumberOfints = 4 - dots.Count;
                if (remainingLength > remainingNumberOfints * 3 ||
                    remainingLength < remainingNumberOfints)
                    return;
                if (dots.Count == 3)
                {
                    if (Valid(s, startIndex, remainingLength))
                    {
                        var temp = "";
                        var last = 0;
                        foreach (var dot in dots)
                        {
                            temp += s.Substring(last, dot) + ".";
                            last += dot;
                        }

                        temp += s.Substring(startIndex);
                        ans.Add(temp);
                    }

                    return;
                }

                for (int curPos = 1; curPos <= 3 && curPos <= remainingLength;
                     ++curPos)
                {
                    dots.Add(curPos);
                    if (Valid(s, startIndex, curPos))
                    {
                        Helper(s, startIndex + curPos, dots, ans);
                    }

                    dots.RemoveAt(dots.Count - 1);
                }
            }

            public IList<string> RestoreIpAddresses(string s)
            {
                var ans = new List<string>();
                Helper(s, 0, new List<int>(), ans);
                return ans;
            }

            /*

        Approach 2: Iterative

            Complexity Analysis
        Let's assume we need to separate the input string into N integers, each integer is at most M digits.
        •	Time complexity: O(M^N⋅N).
        We have (N−1) nested loops and each of them iterates at most M times, so the total number of iterations is at most M^(N−1) .
        In each iteration we split N substrings out to check whether they are valid, each substring's length is at most M, so the time complexity to separate out all of them is O(M⋅N).
        For this question, M = 3, N = 4, so the time complexity is O(1).
        •	Space complexity: O(M⋅N).
        The algorithm saves (N - 1) numbers (the number of digits before each dot) which takes O(N) space. And we need temporary space to save a solution before putting it into the answer list. The length of each solution string is M⋅N+M−1 = O(M⋅N), so the total space complexity is O(M⋅N) if we don't take the output space into consideration.
        For this question, M = 3, N = 4, so the space complexity is O(1).

            */
            public static List<string> RestoreIPAddressesIterative(string str)
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

        }

        /*
        751. IP to CIDR
https://leetcode.com/problems/ip-to-cidr/description/

        */
        public List<string> IpToCIDR(string ip, int n)
        {
            int current = ToInt(ip);
            List<string> result = new List<string>();
            while (n > 0)
            {
                int maxBits = CountTrailingZeros(current);
                int maxAmount = 1 << maxBits;
                int bitValue = 1;
                int count = 0;
                while (bitValue < n && count < maxBits)
                {
                    bitValue <<= 1;
                    ++count;
                }
                if (bitValue > n)
                {
                    bitValue >>= 1;
                    --count;
                }
                result.Add(ToString(current, 32 - count));
                n -= bitValue;
                current += bitValue;
            }
            return result;
        }

        private string ToString(int number, int range)
        {
            //convert every 8 into an integer
            const int WORD_SIZE = 8;
            StringBuilder stringBuilder = new StringBuilder();
            for (int i = 3; i >= 0; --i)
            {
                stringBuilder.Append(((number >> (i * WORD_SIZE)) & 255).ToString());
                stringBuilder.Append(".");
            }
            stringBuilder.Length -= 1;
            stringBuilder.Append("/");
            stringBuilder.Append(range.ToString());
            return stringBuilder.ToString();
        }

        private int ToInt(string ip)
        {
            string[] separators = ip.Split('.');
            int sum = 0;
            for (int i = 0; i < separators.Length; ++i)
            {
                sum *= 256;
                sum += int.Parse(separators[i]);
            }
            return sum;
        }

        private int CountTrailingZeros(int value)
        {
            // This method counts the number of trailing zeros in the binary representation of the integer
            int count = 0;
            while ((value & 1) == 0 && value != 0)
            {
                count++;
                value >>= 1;
            }
            return count;
        }

        /*
        121. Best Time to Buy and Sell Stock
        https://leetcode.com/problems/best-time-to-buy-and-sell-stock/description/

        */
        public class BestTimeToBuyAndSellStockForMaxProfitSol
        {
            /*
            Approach 1: Brute Force
Complexity Analysis
•	Time complexity: O(n^2). Loop runs (n(n−1))/2times.
•	Space complexity: O(1). Only two variables - maxprofit and profit are used.

            */
            public int Naive(int[] prices)
            {
                int maxprofit = 0;
                for (int i = 0; i < prices.Length - 1; i++)
                {
                    for (int j = i + 1; j < prices.Length; j++)
                    {
                        int profit = prices[j] - prices[i];
                        if (profit > maxprofit)
                            maxprofit = profit;
                    }
                }

                return maxprofit;
            }
            /*
            Approach 2: One Pass or Single Loop
Complexity Analysis
Time complexity: O(n). Only a single pass is needed.
Space complexity: O(1). Only two variables are used.

            */
            public int OnePass(int[] prices)
            {
                int minprice = int.MaxValue;
                int maxprofit = 0;
                for (int i = 0; i < prices.Length; i++)
                {
                    if (prices[i] < minprice)
                        minprice = prices[i];
                    else if (prices[i] - minprice > maxprofit)
                        maxprofit = prices[i] - minprice;
                }

                return maxprofit;
            }


        }

        /*
        122. Best Time to Buy and Sell Stock II
        https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/description/	
        */
        public class BestTimeToBuyAndSellStockForMaxProfitIISol
        {
            /*
        
Approach 1: Brute Force
In this case, we simply calculate the profit corresponding to all the possible sets of transactions and find out the maximum profit out of them.
Complexity Analysis
•	Time complexity : O(n^n). Recursive function is called nn times.
•	Space complexity : O(n). Depth of recursion is n.
    
            */
            public int Naive(int[] prices)
            {
                return Calculate(prices, 0);
                int Calculate(int[] prices, int s)
                {
                    if (s >= prices.Length)
                        return 0;
                    int max = 0;
                    for (int start = s; start < prices.Length; start++)
                    {
                        int maxprofit = 0;
                        for (int i = start + 1; i < prices.Length; i++)
                        {
                            if (prices[start] < prices[i])
                            {
                                int profit =
                                    Calculate(prices, i + 1) + prices[i] - prices[start];
                                if (profit > maxprofit)
                                    maxprofit = profit;
                            }
                        }

                        if (maxprofit > max)
                            max = maxprofit;
                    }

                    return max;
                }
            }
            /*
    
Approach 2: Peak Valley Approach with Two Pass
     Complexity Analysis
Time complexity : O(n). Single pass.
Space complexity : O(1). Constant space required.

   
            */
            public int PeakValleyWithTwoPass(int[] prices)
            {
                int i = 0;
                int valley = prices[0];
                int peak = prices[0];
                int maxprofit = 0;
                while (i < prices.Length - 1)
                {
                    while (i < prices.Length - 1 && prices[i] >= prices[i + 1]) i++;
                    valley = prices[i];
                    while (i < prices.Length - 1 && prices[i] <= prices[i + 1]) i++;
                    peak = prices[i];
                    maxprofit += peak - valley;
                }

                return maxprofit;
            }

            /*
            Approach 3: Peak Valley with Simple One Pass
            Complexity Analysis
Time complexity : O(n). Single pass.
Space complexity: O(1). Constant space needed.


            */
            public int PeakValleyWithSinglePass(int[] prices)
            {
                int maxprofit = 0;
                for (int i = 1; i < prices.Length; i++)
                {
                    if (prices[i] > prices[i - 1])
                        maxprofit += prices[i] - prices[i - 1];
                }

                return maxprofit;
            }

        }



        /*
        123. Best Time to Buy and Sell Stock III
        https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/description/

        */
        public class BestTimeToBuyAndSellStockForMaxProfitIIIAtMostTwoTransactSol
        {
            /*            
Approach 1: Bidirectional Dynamic Programming
Complexity
•	Time Complexity: O(N) where N is the length of the input sequence, since we have two iterations of length N.
•	Space Complexity: O(N) for the two arrays that we keep in the algorithm.


            */
            public int BidirectionalDP(int[] prices)
            {
                int length = prices.Length;
                if (length <= 1)
                    return 0;
                int leftMin = prices[0];
                int rightMax = prices[length - 1];
                int[] leftProfits = new int[length];
                int[] rightProfits = new int[length + 1];
                for (var l = 1; l < length; ++l)
                {
                    leftProfits[l] = Math.Max(leftProfits[l - 1], prices[l] - leftMin);
                    leftMin = Math.Min(leftMin, prices[l]);
                    int r = length - 1 - l;
                    rightProfits[r] =
                        Math.Max(rightProfits[r + 1], rightMax - prices[r]);
                    rightMax = Math.Max(rightMax, prices[r]);
                }

                var maxProfit = 0;
                for (var i = 0; i < length; ++i)
                    maxProfit =
                        Math.Max(maxProfit, leftProfits[i] + rightProfits[i + 1]);
                return maxProfit;
            }
            /*
            Approach 2: One-pass Simulation
Complexity
•	Time Complexity: O(N), where N is the length of the input sequence.
•	Space Complexity: O(1), only constant memory is required, which is invariant from the input sequence.

            */
            public int OnePassSimulation(int[] prices)
            {
                int t1Cost = Int32.MaxValue, t2Cost = Int32.MaxValue;
                int t1Profit = 0, t2Profit = 0;
                foreach (int price in prices)
                {
                    // the maximum profit if only one transaction is allowed
                    t1Cost = Math.Min(t1Cost, price);
                    t1Profit = Math.Max(t1Profit, price - t1Cost);
                    // reinvest the gained profit in the second transaction
                    t2Cost = Math.Min(t2Cost, price - t1Profit);
                    t2Profit = Math.Max(t2Profit, price - t2Cost);
                }

                return t2Profit;
            }














        }



        /*
        188. Best Time to Buy and Sell Stock IV
        https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iv/description/
        */
        public class BestTimeToBuyAndSellStockForMaxProfitIVAtMostKTransactSol
        {

            /*
            Approach 1: Dynamic Programming
            Complexity Analysis
•	Time Complexity: O(nk) if k⋅2≤n, O(n) if k⋅2>n, where n is the length of the prices sequence since we have two for-loop.
•	Space Complexity: O(nk) without state-compressed, and O(k) with state-compressed, where n is the length of the prices sequence.

            */
            public int DP(int k, int[] prices)
            {
                int n = prices.Length;

                // Solve special cases
                if (n <= 0 || k <= 0)
                {
                    return 0;
                }

                if (k * 2 >= n)
                {
                    int tmpRes = 0;
                    for (int i = 1; i < n; i++)
                    {
                        tmpRes += Math.Max(0, prices[i] - prices[i - 1]);
                    }
                    return tmpRes;
                }

                // dp[i][used_k][ishold] = balance
                // ishold: 0 nothold, 1 hold
                int[][][] dp = new int[n][][];

                // initialize the array with -inf
                // we use -1e9 here to prevent overflow
                for (int i = 0; i < n; i++)
                {
                    for (int j = 0; j <= k; j++)
                    {
                        dp[i][j][0] = -1000000000;
                        dp[i][j][1] = -1000000000;
                    }
                }

                // set starting value
                dp[0][0][0] = 0;
                dp[0][1][1] = -prices[0];

                // fill the array
                for (int i = 1; i < n; i++)
                {
                    for (int j = 0; j <= k; j++)
                    {
                        // transition equation
                        dp[i][j][0] = Math.Max(
                            dp[i - 1][j][0],
                            dp[i - 1][j][1] + prices[i]
                        );
                        // you can't hold stock without any transaction
                        if (j > 0)
                        {
                            dp[i][j][1] = Math.Max(
                                dp[i - 1][j][1],
                                dp[i - 1][j - 1][0] - prices[i]
                            );
                        }
                    }
                }

                int res = 0;
                for (int j = 0; j <= k; j++)
                {
                    res = Math.Max(res, dp[n - 1][j][0]);
                }

                return res;
            }

            /*
            Approach 2: Merging
            Complexity Analysis
•	Time Complexity: O(n(n−k)) if (k/2)≤n , O(n) if (k/2)>n, where n is the length of the price sequence. The maximum size of transactions is O(n), and we need O(n−k) iterations.
•	Space Complexity: O(n), since we need a list to store transactions.

            */
            public int Merging(int k, int[] prices)
            {
                int n = prices.Length;

                // solve special cases
                if (n <= 0 || k <= 0)
                {
                    return 0;
                }

                // find all consecutively increasing subsequence
                List<int[]> transactions = new List<int[]>();
                int start = 0;
                int end = 0;
                for (int i = 1; i < n; i++)
                {
                    if (prices[i] >= prices[i - 1])
                    {
                        end = i;
                    }
                    else
                    {
                        if (end > start)
                        {
                            int[] t = { start, end };
                            transactions.Add(t);
                        }
                        start = i;
                    }
                }
                if (end > start)
                {
                    int[] t = { start, end };
                    transactions.Add(t);
                }

                while (transactions.Count > k)
                {
                    // check delete loss
                    int delete_index = 0;
                    int min_delete_loss = int.MaxValue;
                    for (int i = 0; i < transactions.Count; i++)
                    {
                        int[] t = transactions[i];
                        int profit_loss = prices[t[1]] - prices[t[0]];
                        if (profit_loss < min_delete_loss)
                        {
                            min_delete_loss = profit_loss;
                            delete_index = i;
                        }
                    }

                    // check merge loss
                    int merge_index = 0;
                    int min_merge_loss = int.MaxValue; ;
                    for (int i = 1; i < transactions.Count; i++)
                    {
                        int[] t1 = transactions[i - 1];
                        int[] t2 = transactions[i];
                        int profit_loss = prices[t1[1]] - prices[t2[0]];
                        if (profit_loss < min_merge_loss)
                        {
                            min_merge_loss = profit_loss;
                            merge_index = i;
                        }
                    }

                    // delete or merge
                    if (min_delete_loss <= min_merge_loss)
                    {
                        transactions.RemoveAt(delete_index);
                    }
                    else
                    {
                        int[] t1 = transactions[merge_index - 1];
                        int[] t2 = transactions[merge_index];
                        t1[1] = t2[1];
                        transactions.RemoveAt(merge_index);
                    }
                }

                int res = 0;
                foreach (int[] t in transactions)
                {
                    res += prices[t[1]] - prices[t[0]];
                }

                return res;
            }

        }


        /*
        309. Best Time to Buy and Sell Stock with Cooldown
        https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/description/

        */
        public class BestTimeToBuyAndSellStockForMaxProfitWithColldownSol
        {
            /*         
 Approach 1: Dynamic Programming with State Machine
Complexity Analysis
•	Time Complexity: O(N) where N is the length of the input price list.
o	We have one loop over the input list, and the operation within one iteration takes constant time.
•	Space Complexity: O(1), constant memory is used regardless the size of the input.


            */
            public int StateMachineDP(int[] prices)
            {

                int sold = int.MinValue, held = int.MinValue, reset = 0;

                foreach (int price in prices)
                {
                    int preSold = sold;

                    sold = held + price;
                    held = Math.Max(held, reset - price);
                    reset = Math.Max(reset, preSold);
                }

                return Math.Max(sold, reset);
            }

            /*
            Approach 2: Yet-Another Dynamic Programming
Complexity Analysis
•	Time Complexity: O(N^2) where N is the length of the price list.
o	As one can see, we have nested loops over the price list. The number of iterations in the outer loop is N. The number of iterations in the inner loop varies from 1 to N. Therefore, the total number of iterations that we perform is ∑i=1 toN where i=(N⋅(N+1))/2.
o	As a result, the overall time complexity of the algorithm is O(N^2).
•	Space Complexity: O(N) where N is the length of the price list.
o	We allocated an array to hold all the values for our target function MP(i).

            */
            public int DP2(int[] prices)
            {
                int[] MP = new int[prices.Length + 2];

                for (int i = prices.Length - 1; i >= 0; i--)
                {
                    int C1 = 0;
                    // Case 1). buy and sell the stock
                    for (int sell = i + 1; sell < prices.Length; sell++)
                    {
                        int profit = (prices[sell] - prices[i]) + MP[sell + 2];
                        C1 = Math.Max(profit, C1);
                    }

                    // Case 2). do no transaction with the stock p[i]
                    int C2 = MP[i + 1];

                    // wrap up the two cases
                    MP[i] = Math.Max(C1, C2);
                }
                return MP[0];
            }
        }


        /*
        714. Best Time to Buy and Sell Stock with Transaction Fee
        https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/description/

        */
        public class BestTimeToBuyAndSellStockForMaxProfitWithTrasactFeeSol
        {
            /*
            Approach 1: Dynamic Programming
            Complexity Analysis
Let n be the length of the input array prices.
•	Time complexity: O(n)
o	We iterate from day 1 to day n - 1, which contains n - 1 steps.
o	At each step, we update free[i] and hold[i] which takes O(1).
•	Space complexity: O(n)
o	We create two arrays of length n to record the maximum profit with two status on each day.

            */
            public int DP(int[] prices, int fee)
            {
                int n = prices.Length;
                int[] free = new int[n], hold = new int[n];

                // In order to hold a stock on day 0, we have no other choice but to buy it for prices[0].
                hold[0] = -prices[0];

                for (int i = 1; i < n; i++)
                {
                    hold[i] = Math.Max(hold[i - 1], free[i - 1] - prices[i]);
                    free[i] = Math.Max(free[i - 1], hold[i - 1] + prices[i] - fee);
                }

                return free[n - 1];
            }
            /*
            Approach 2: Space-Optimized Dynamic Programming
Complexity Analysis
Let n be the length of the input array prices.
•	Time complexity: O(n)
o	We iterate from day 1 to day n - 1, which contains n - 1 steps.
o	At each step, we update free and hold which takes O(1).
•	Space complexity: O(1)
o	We only need to update three parameters tmp, free and hold.

            */
            public int DPSpaceOptimal(int[] prices, int fee)
            {
                int n = prices.Length;
                int free = 0, hold = -prices[0];

                for (int i = 1; i < n; i++)
                {
                    int tmp = hold;
                    hold = Math.Max(hold, free - prices[i]);
                    free = Math.Max(free, tmp + prices[i] - fee);
                }

                return free;
            }
        }


        /*   
        2291. Maximum Profit From Trading Stocks
    https://leetcode.com/problems/maximum-profit-from-trading-stocks/description/

        */
        /*
        Knapsack
        •	Time: O(mn), m = len(present), n = budget
        */
        public int Knapsack(IList<int> present, IList<int> future, int budget)
        {
            int[] dp = new int[budget + 1];

            for (int i = 0; i < present.Count; i++)
            {
                int p = present[i];
                int f = future[i];
                for (int j = budget; j >= p; j--)
                {
                    dp[j] = Math.Max(dp[j], dp[j - p] + f - p);
                }
            }

            return dp[budget];
        }

        /*
        139. Word Break		
https://leetcode.com/problems/word-break/description/

        */
        public class WordBreakSol
        {
            /*
            Approach 1: Breadth-First Search
            Complexity Analysis
Given n as the length of s, m as the length of wordDict, and k as the average length of the words in wordDict,
•	Time complexity: O(n3+m⋅k)
There are O(n) nodes. Because of seen, we never visit a node more than once. At each node, we iterate over the nodes in front of the current node, of which there are O(n). For each node end, we create a substring, which also costs O(n).
Therefore, handling a node costs O(n2), so the BFS could cost up to O(n3). Finally, we also spent O(m⋅k) to create the set words.
•	Space complexity: O(n+m⋅k)
We use O(n) space for queue and seen. We use O(m⋅k) space for the set words.

            */
            public static bool BFS(string s, IList<string> wordDict)
            {
                HashSet<string> words = new HashSet<string>(wordDict);
                Queue<int> queue = new Queue<int>();
                bool[] seen = new bool[s.Length + 1];
                queue.Enqueue(0);
                while (queue.Count != 0)
                {
                    int start = queue.Dequeue();
                    if (start == s.Length)
                    {
                        return true;
                    }

                    for (int end = start + 1; end <= s.Length; end++)
                    {
                        if (seen[end])
                        {
                            continue;
                        }

                        if (words.Contains(s.Substring(start, end - start)))
                        {
                            queue.Enqueue(end);
                            seen[end] = true;
                        }
                    }
                }

                return false;
            }
            /*
            Approach 2: Top-Down Dynamic Programming
Complexity Analysis
Given n as the length of s, m as the length of wordDict, and k as the average length of the words in wordDict,
•	Time complexity: O(n⋅m⋅k)
There are n states of dp(i). Because of memoization, we only calculate each state once. To calculate a state, we iterate over m words, and for each word perform some substring operations which costs O(k). Therefore, calculating a state costs O(m⋅k), and we need to calculate O(n) states.
•	Space complexity: O(n)
The data structure we use for memoization and the recursion call stack can use up to O(n) space.

            */
            public static bool TopDownDP(string s, IList<string> wordDict)
            {
                int[] memo = new int[s.Length];
                Array.Fill(memo, -1);
                return IsValid(s.Length - 1);

                bool IsValid(int i)
                {
                    if (i < 0)
                    {
                        return true;
                    }

                    if (memo[i] != -1)
                    {
                        return memo[i] == 1;
                    }

                    foreach (string word in wordDict)
                    {
                        // Handle out of bounds case
                        if (i - word.Length + 1 < 0)
                        {
                            continue;
                        }

                        if (s.Substring(i - word.Length + 1, word.Length) == word &&
                            IsValid(i - word.Length))
                        {
                            memo[i] = 1;
                            return true;
                        }
                    }

                    memo[i] = 0;
                    return false;
                }

            }
            /*
            Approach 3: Bottom-Up Dynamic Programming
Complexity Analysis
Given n as the length of s, m as the length of wordDict, and k as the average length of the words in wordDict,
•	Time complexity: O(n⋅m⋅k)
The logic behind the time complexity is identical to the previous approach. It costs us O(m⋅k) to calculate each state, and we calculate O(n) states in total.
•	Space complexity: O(n)
We use an array dp of length n.	

            */
            public static bool BottomUpDP(string s, IList<string> wordDict)
            {
                bool[] dp = new bool[s.Length];
                for (int i = 0; i < s.Length; i++)
                {
                    foreach (string word in wordDict)
                    {
                        // Handle out of bounds case
                        if (i < word.Length - 1)
                        {
                            continue;
                        }

                        if (i == word.Length - 1 || dp[i - word.Length])
                        {
                            if (s.Substring(i - word.Length + 1, word.Length) == word)
                            {
                                dp[i] = true;
                                break;
                            }
                        }
                    }
                }

                return dp[s.Length - 1];
            }
            /*
            Approach 4: Trie Optimization
Complexity Analysis
Given n as the length of s, m as the length of wordDict, and k as the average length of the words in wordDict,
•	Time complexity: O(n^2+m⋅k)
Building the trie involves iterating over all characters of all words. This costs O(m⋅k).
Once we build the trie, we calculate dp. For each i, we iterate over all the indices after i. We have a basic nested for loop which costs O(n2) to handle all dp[i].
•	Space complexity: O(n+m⋅k)
The dp array takes O(n) space. The trie can have up to m⋅k nodes in it.

            */

            public bool TrieOptimal(string s, IList<string> wordDict)
            {
                TrieNode root = new TrieNode();
                foreach (string word in wordDict)
                {
                    TrieNode curr = root;
                    foreach (char c in word)
                    {
                        if (!curr.children.ContainsKey(c))
                        {
                            curr.children[c] = new TrieNode();
                        }

                        curr = curr.children[c];
                    }

                    curr.isWord = true;
                }

                bool[] dp = new bool[s.Length];
                for (int i = 0; i < s.Length; i++)
                {
                    if (i == 0 || dp[i - 1])
                    {
                        TrieNode curr = root;
                        for (int j = i; j < s.Length; j++)
                        {
                            char c = s[j];
                            if (!curr.children.ContainsKey(c))
                            {
                                // No words exist
                                break;
                            }

                            curr = curr.children[c];
                            if (curr.isWord)
                            {
                                dp[j] = true;
                            }
                        }
                    }
                }

                return dp[s.Length - 1];



            }
            public class TrieNode
            {
                public bool isWord;
                public Dictionary<char, TrieNode> children;

                public TrieNode()
                {
                    this.children = new Dictionary<char, TrieNode>();
                }
            }


            /*
Approach 5: A Different DP
Complexity Analysis
Given n as the length of s, m as the length of wordDict, and k as the average length of the words in wordDict,
•	Time complexity: O(n^3+m⋅k)
First, we spend O(m⋅k) to convert wordDict into a set. Then we have a nested loop over n, which iterates O(n^2) times. For each iteration, we have a substring operation which could cost up to O(n). Thus this nested loop costs O(n^3).
•	Space complexity: O(n+m⋅k)
The dp array takes O(n) space. The set words takes up O(m⋅k) space.	

            */
            public bool VariationDP(string s, IList<string> wordDict)
            {
                HashSet<string> words = new HashSet<string>(wordDict);
                bool[] dp = new bool[s.Length + 1];
                dp[0] = true;
                for (int i = 1; i <= s.Length; i++)
                {
                    for (int j = 0; j < i; j++)
                    {
                        if (dp[j] && words.Contains(s.Substring(j, i - j)))
                        {
                            dp[i] = true;
                            break;
                        }
                    }
                }

                return dp[s.Length];
            }

        }

        /* 140. Word Break II
        https://leetcode.com/problems/word-break-ii/description/
         */

        public class WordBreakIISol
        {
            /*
            Approach 1: Backtracking
Complexity Analysis
Let n be the length of the input string.
•	Time complexity: O(n⋅(2^n))
The algorithm explores all possible ways to break the string into words. In the worst case, where each character can be treated as a word, the recursion tree has 2n leaf nodes, resulting in an exponential time complexity. For each leaf node, O(n) work is performed, so the overall complexity is O(n⋅(2^n)).
•	Space complexity: O(2^n)
The recursion stack can grow up to a depth of n, where each recursive call consumes additional space for storing the current state.
Since each position in the string can be a split point or not, and for n positions, there are (2^n) possible combinations of splits. Thus, in the worst case, each combination generates a different sentence that needs to be stored, leading to exponential space complexity.

            */
            public IList<string> UsingBacktracking(string inputString, IList<string> wordDictionary)
            {
                // Convert wordDictionary to a HashSet for O(1) lookups
                HashSet<string> wordSet = new HashSet<string>(wordDictionary);
                List<string> results = new List<string>();
                // Start the backtracking process
                Backtrack(inputString, wordSet, new StringBuilder(), results, 0);
                return results;
            }

            private void Backtrack(
                string inputString,
                HashSet<string> wordSet,
                StringBuilder currentSentence,
                List<string> results,
                int startIndex
            )
            {
                // If we've reached the end of the string, add the current sentence to results
                if (startIndex == inputString.Length)
                {
                    results.Add(currentSentence.ToString().Trim());
                    return;
                }

                // Iterate over possible end indices
                for (int endIndex = startIndex + 1; endIndex <= inputString.Length; endIndex++)
                {
                    string word = inputString.Substring(startIndex, endIndex - startIndex);
                    // If the word is in the set, proceed with backtracking
                    if (wordSet.Contains(word))
                    {
                        int currentLength = currentSentence.Length;
                        currentSentence.Append(word).Append(" ");
                        // Recursively call backtrack with the new end index
                        Backtrack(inputString, wordSet, currentSentence, results, endIndex);
                        // Reset currentSentence to its original length
                        currentSentence.Length = currentLength;
                    }
                }

            }
            /*
            Approach 2: Dynamic Programming - Memoization
            Complexity Analysis
           Let n be the length of the input string.
           •	Time complexity: O(n⋅(2^n))
           While memoization avoids redundant computations, it does not change the overall number of subproblems that need to be solved. In the worst case, there are still unique (2^n) possible substrings that need to be explored, leading to an exponential time complexity. For each subproblem, O(n) work is performed, so the overall complexity is O(n⋅(2^n)).
           •	Space complexity: O(n⋅(2^n))
           The recursion stack can grow up to a depth of n, where each recursive call consumes additional space for storing the current state.
           The memoization map needs to store the results for all possible substrings, which can be up to (2^n) substrings of size n in the worst case, resulting in an exponential space complexity.


           */
            // Main function to break the string into words
            public List<string> DPMemoDFS(string inputString, List<string> wordDictionary)
            {
                HashSet<string> wordSet = new HashSet<string>(wordDictionary);
                Dictionary<string, List<string>> memoization = new Dictionary<string, List<string>>();
                return DepthFirstSearch(inputString, wordSet, memoization);
            }

            // Depth-first search function to find all possible word break combinations
            private List<string> DepthFirstSearch(
                string remainingString,
                HashSet<string> wordSet,
                Dictionary<string, List<string>> memoization
            )
            {
                // Check if result for this substring is already memoized
                if (memoization.ContainsKey(remainingString))
                {
                    return memoization[remainingString];
                }

                // Base case: when the string is empty, return a list containing an empty string
                if (remainingString == string.Empty) return new List<string> { string.Empty };
                List<string> results = new List<string>();
                for (int index = 1; index <= remainingString.Length; ++index)
                {
                    string currentWord = remainingString.Substring(0, index);
                    // If the current substring is a valid word
                    if (wordSet.Contains(currentWord))
                    {
                        foreach (string nextWord in DepthFirstSearch(
                            remainingString.Substring(index),
                            wordSet,
                            memoization
                        ))
                        {
                            // Append current word and next word with space in between if next word exists
                            results.Add(currentWord + (nextWord == string.Empty ? string.Empty : " ") + nextWord);
                        }
                    }
                }
                // Memoize the results for the current substring
                memoization[remainingString] = results;
                return results;
            }
            /*
            Approach 3: Dynamic Programming - Tabulation
Complexity Analysis
Let n be the length of the input string.
•	Time complexity: O(n⋅(2^n))
Similar to memoization, the tabulation approach still needs to explore all possible substrings, which can be up to (2^n) in the worst case, leading to an exponential time complexity. O(n) work is performed to explore each substring, so the overall complexity is O(n⋅(2^n)).
•	Space complexity: O(n⋅(2^n))
The dynamic programming table or map needs to store the valid sentences for all possible starting indices, which can be up to (2^n) strings of size n in the worst case, resulting in an exponential space complexity.

            */
            public List<string> DPTabulation(string inputString, List<string> wordDictionary)
            {
                // Dictionary to store results of subproblems
                Dictionary<int, List<string>> dynamicProgramming = new Dictionary<int, List<string>>();

                // Iterate from the end of the string to the beginning
                for (int startIndex = inputString.Length; startIndex >= 0; startIndex--)
                {
                    // List to store valid sentences starting from startIndex
                    List<string> validSentences = new List<string>();

                    // Iterate from startIndex to the end of the string
                    for (int endIndex = startIndex; endIndex < inputString.Length; endIndex++)
                    {
                        // Extract substring from startIndex to endIndex
                        string currentWord = inputString.Substring(startIndex, endIndex - startIndex + 1);

                        // Check if the current substring is a valid word
                        if (IsWordInDictionary(currentWord, wordDictionary))
                        {
                            // If it's the last word, add it as a valid sentence
                            if (endIndex == inputString.Length - 1)
                            {
                                validSentences.Add(currentWord);
                            }
                            else
                            {
                                // If it's not the last word, append it to each sentence formed by the remaining substring
                                List<string> sentencesFromNextIndex;
                                if (dynamicProgramming.TryGetValue(endIndex + 1, out sentencesFromNextIndex))
                                {
                                    foreach (string sentence in sentencesFromNextIndex)
                                    {
                                        validSentences.Add(currentWord + " " + sentence);
                                    }
                                }
                            }
                        }
                    }
                    // Store the valid sentences in dynamicProgramming
                    dynamicProgramming[startIndex] = validSentences;
                }
                // Return the sentences formed from the entire string
                return dynamicProgramming.ContainsKey(0) ? dynamicProgramming[0] : new List<string>();
            }

            // Helper function to check if a word is in the word dictionary
            private bool IsWordInDictionary(string word, List<string> wordDictionary)
            {
                return wordDictionary.Contains(word);
            }
            /*
            Approach 4: Trie Optimization
           Complexity Analysis
Let n be the length of the input string.
•	Time complexity: O(n⋅(2^n))
Even though the trie-based approach uses an efficient data structure for word lookup, it still needs to explore all possible ways to break the string into words. In the worst case, there are (2^n)unique possible partitions, leading to an exponential time complexity. O(n) work is performed for each partition, so the overall complexity is O(n⋅(2^n)).
•	Space complexity: O(n⋅(2^n))
The trie data structure itself can have a maximum of (2^n) nodes in the worst case, where each character in the string represents a separate word. Additionally, the tabulation map used in this approach can also store up to (2^n) strings of size n, resulting in an overall exponential space complexity.
 
            */
            public IList<string> UsingTrie(string s, IList<string> wordDict)
            {
                // Build the Trie from the word dictionary
                Trie trie = new Trie();
                foreach (string word in wordDict)
                {
                    trie.Insert(word);
                }

                // Map to store results of subproblems
                Dictionary<int, List<string>> dp = new Dictionary<int, List<string>>();

                // Iterate from the end of the string to the beginning
                for (int startIdx = s.Length; startIdx >= 0; startIdx--)
                {
                    // List to store valid sentences starting from startIdx
                    List<string> validSentences = new List<string>();

                    // Initialize current node to the root of the trie
                    TrieNode currentNode = trie.Root;

                    // Iterate from startIdx to the end of the string
                    for (int endIdx = startIdx; endIdx < s.Length; endIdx++)
                    {
                        char c = s[endIdx];
                        int index = c - 'a';

                        // Check if the current character exists in the trie
                        if (currentNode.Children[index] == null)
                        {
                            break;
                        }

                        // Move to the next node in the trie
                        currentNode = currentNode.Children[index];

                        // Check if we have found a valid word
                        if (currentNode.IsEnd)
                        {
                            string currentWord = s.Substring(startIdx, endIdx - startIdx + 1);

                            // If it's the last word, add it as a valid sentence
                            if (endIdx == s.Length - 1)
                            {
                                validSentences.Add(currentWord);
                            }
                            else
                            {
                                // If it's not the last word, append it to each sentence formed by the remaining substring
                                if (dp.TryGetValue(endIdx + 1, out List<string> sentencesFromNextIndex))
                                {
                                    foreach (string sentence in sentencesFromNextIndex)
                                    {
                                        validSentences.Add(currentWord + " " + sentence);
                                    }
                                }
                            }
                        }
                    }

                    // Store the valid sentences in dp
                    dp[startIdx] = validSentences;
                }

                // Return the sentences formed from the entire string
                return dp.TryGetValue(0, out List<string> result) ? result : new List<string>();
            }
            public class TrieNode
            {
                public bool IsEnd;
                public TrieNode[] Children;

                public TrieNode()
                {
                    IsEnd = false;
                    Children = new TrieNode[26];
                }
            }

            public class Trie
            {
                public TrieNode Root;

                public Trie()
                {
                    Root = new TrieNode();
                }

                public void Insert(string word)
                {
                    TrieNode node = Root;
                    foreach (char c in word)
                    {
                        int index = c - 'a';
                        if (node.Children[index] == null)
                        {
                            node.Children[index] = new TrieNode();
                        }
                        node = node.Children[index];
                    }
                    node.IsEnd = true;
                }
            }

        }

        /*
        2534. Time Taken to Cross the Door
        https://leetcode.com/problems/time-taken-to-cross-the-door/description/
        https://algo.monster/liteproblems/2534
        */
        public class TimeTakenToCroosDoorSol
        {
            /*
            Approach: Two Queues
Complexity
•	Time complexity: O(n).
•	Space complexity: O(n).

            */
            public int[] TWoQueues(int[] arrival, int[] state)
            {
                int[] result = new int[arrival.Length];
                //timer
                int currentTime = 0;
                //build 2 queues;
                Queue<int> enterQueue = new Queue<int>();
                Queue<int> exitQueue = new Queue<int>();
                // Handle the 4th rule:If multiple persons want to go in the same direction, the person with the smallest index goes first.
                for (int i = 0; i < state.Length; i++)
                {
                    if (state[i] == 0)
                    {
                        enterQueue.Enqueue(i);
                    }
                    else
                    {
                        exitQueue.Enqueue(i);
                    }
                }
                // handle the 1st rule at time = 0;
                int previousAction = 1;
                // timer start
                while (enterQueue.Count > 0 && exitQueue.Count > 0)
                {
                    //Two or more person at the door, handle 2nd and 3rd rule;
                    if (arrival[enterQueue.Peek()] <= currentTime && arrival[exitQueue.Peek()] <= currentTime)
                    {
                        if (previousAction == 0)
                        {
                            int index = enterQueue.Dequeue();
                            result[index] = currentTime;
                        }
                        else
                        {
                            int index = exitQueue.Dequeue();
                            result[index] = currentTime;
                        }
                        //Only one person at the door to enter;
                    }
                    else if (arrival[enterQueue.Peek()] <= currentTime && arrival[exitQueue.Peek()] > currentTime)
                    {
                        int index = enterQueue.Dequeue();
                        result[index] = currentTime;
                        previousAction = 0;
                        //Only one person at the door to exit;
                    }
                    else if (arrival[enterQueue.Peek()] > currentTime && arrival[exitQueue.Peek()] <= currentTime)
                    {
                        int index = exitQueue.Dequeue();
                        result[index] = currentTime;
                        previousAction = 1;
                        //No one at the door now, handle the 1st rule;
                    }
                    else
                    {
                        previousAction = 1;
                    }
                    currentTime++;
                }
                //clear queues
                while (enterQueue.Count > 0)
                {
                    int index = enterQueue.Dequeue();
                    result[index] = Math.Max(arrival[index], currentTime);
                    currentTime = Math.Max(arrival[index], currentTime) + 1;
                }
                while (exitQueue.Count > 0)
                {
                    int index = exitQueue.Dequeue();
                    result[index] = Math.Max(arrival[index], currentTime);
                    currentTime = Math.Max(arrival[index], currentTime) + 1;
                }

                return result;
            }

        }

        /*
        2332. The Latest Time to Catch a Bus
        https://leetcode.com/problems/the-latest-time-to-catch-a-bus/description/
        https://algo.monster/liteproblems/2332

        */
        public class LatestTimeCatchTheBusSol
        {

            /*
            Approach: Sorting

            Time and Space Complexity
            Time Complexity
            The time complexity of the code is determined by the sorting of the buses and passengers lists, and the iterations over these lists.
            1.	buses.sort() sorts the list of buses. Sorting a list of n elements has a time complexity of O(n log n), where n is the number of buses in this context.
            2.	passengers.sort() sorts the list of passengers. The sorting has a time complexity of O(m log m), where m is the number of passengers.
            3.	The for loop iterates over each bus – this is O(n) where n is the number of buses.
            4.	The nested while loop iterates over the passengers, but it only processes each passenger once in total, not once per bus. Hence, the total number of inner loop iterations is O(m) across all iterations of the outer loop, where m is the total number of passengers.
            Adding these up, we get a time complexity of O(n log n) + O(m log m) + O(n) + O(m). Since the O(n log n) and O(m log m) terms will be dominant for large n and m, we can simplify this to O(n log n + m log m).
            Space Complexity
            The space complexity is determined by the additional memory used by the program.
            1.	The sorting algorithms for both buses and passengers lists typically have a space complexity of O(1) if implemented as an in-place sort such as Timsort (which is the case in Python's sort() function).
            2.	The additional variables c, j, and ans use constant space, which adds a space complexity of O(1).
            Thus, when not considering the space taken up by the input, the overall space complexity of the code would be O(1). However, if considering the space used by the inputs themselves, we must acknowledge that the lists buses and passengers use O(n + m) space.
            Therefore, the total space complexity, considering input space, is O(n + m).

            */
            // Method to find the latest time you can catch the bus without modifying method names as per guidelines.
            public int Sorting(int[] buses, int[] passengers, int capacity)
            {
                // Sort the buses and passengers to process them in order.
                Array.Sort(buses);
                Array.Sort(passengers);

                // Passenger index and current capacity initialization
                int passengerIndex = 0, currentCapacity = 0;

                // Iterate through each bus
                foreach (int busTime in buses)
                {
                    // Reset capacity for the new bus
                    currentCapacity = capacity;

                    // Load passengers until the bus is either full or all waiting passengers have boarded.
                    while (currentCapacity > 0 && passengerIndex < passengers.Length && passengers[passengerIndex] <= busTime)
                    {
                        currentCapacity--;
                        passengerIndex++;
                    }
                }

                // Decrement to get the last passenger's time or the bus's latest time if it's not full
                passengerIndex--;

                // Determine the latest time that you can catch the bus
                int latestTime;

                // If there is capacity left in the last bus, the latest time is the last bus's departure time.
                // Otherwise, it's the time just before the last passenger boarded.
                if (currentCapacity > 0)
                {
                    latestTime = buses[buses.Length - 1];
                }
                else
                {
                    latestTime = passengers[passengerIndex];
                }

                // Ensure that the latest time is not the same as any passenger's arrival time.
                while (passengerIndex >= 0 && latestTime == passengers[passengerIndex])
                {
                    latestTime--;
                    passengerIndex--;
                }

                // Return the latest time you can catch the bus
                return latestTime;
            }
        }

        /*
        2431. Maximize Total Tastiness of Purchased Fruits
        https://leetcode.com/problems/maximize-total-tastiness-of-purchased-fruits/description/
        https://algo.monster/liteproblems/2431
        */
        public class MaxTastinessSol
        {
            /*
            Approach: using a combination of recursive depth-first search (DFS) and dynamic programming (DP) with memoization

Time and Space Complexity
The given Python code defines a recursive function with memoization to solve a variation of the knapsack problem, where we are trying to maximize the tastiness of items chosen under certain constraints on the price and available coupons.
Time Complexity
The time complexity of the code is dictated by the number of states that need to be computed, which is determined by the number of decisions for each item, the range of maxAmount (denoted as M), and the number of coupons maxCoupons (denoted as C). The function dfs is called with different states represented by a combination of current item index i, remaining amount j, and remaining coupons k.
For each item, there are three choices: skip the item, take the item without a coupon, or take the item with a coupon (if available). Since we only move to the next item in each recursive call, there are N levels of recursion, where N is the total number of items.
There is a unique state for each combination of (i, j, k). Since i can be in the range [0, N], j can take on values from [0, M], and k from [0, C], the number of possible states is roughly N * M * C.
The recursion is memoized to ensure that each state is computed at most once. Therefore, the time complexity is O(N*M*C).
Space Complexity
The space complexity of the code is governed by the storage required for:
1.	The memoization cache, which needs to store the result for each unique state (i, j, k), thus requiring a space complexity of O(N*M*C).
2.	The call stack for the recursion, which at maximum depth will be O(N), as we have N levels of recursion in the worst case.
Thus, the overall space complexity of the code is also O(N*M*C) due to the cache size dominating the recursive call stack.

            */
            private int[][][] memo; // 3D memoization array to store the results of subproblems
            private int[] itemPrices;   // Array to store prices of items
            private int[] itemTastinessValues; // Array to store tastiness values of items
            private int totalItemCount;  // The number of items

            public int DPMemoWithDFSRec(int[] prices, int[] tastinessValues, int maxAmount, int maxCoupons)
            {
                totalItemCount = prices.Length;
                itemPrices = prices;
                itemTastinessValues = tastinessValues;
                memo = new int[totalItemCount][][]; // Initialize 3D array

                // Call the recursive function starting at the first item, with full budget, and all coupons available
                return Dfs(0, maxAmount, maxCoupons);
            }

            private int Dfs(int currentItemIndex, int remainingAmount, int remainingCoupons)
            {
                // Base case: when all items have been considered
                if (currentItemIndex == totalItemCount)
                {
                    return 0;
                }

                // Return the stored result if this subproblem has already been computed
                if (memo[currentItemIndex][remainingAmount][remainingCoupons] != 0)
                {
                    return memo[currentItemIndex][remainingAmount][remainingCoupons];
                }

                // Case 1: Skip the current item and go to the next
                int maxTastinessValue = Dfs(currentItemIndex + 1, remainingAmount, remainingCoupons);

                // Case 2: Buy the current item without a coupon if enough amount remains
                if (remainingAmount >= itemPrices[currentItemIndex])
                {
                    maxTastinessValue = Math.Max(maxTastinessValue, Dfs(currentItemIndex + 1, remainingAmount - itemPrices[currentItemIndex], remainingCoupons) + itemTastinessValues[currentItemIndex]);
                }

                // Case 3: Buy the current item with a coupon if a coupon and enough amount remain
                if (remainingAmount >= itemPrices[currentItemIndex] / 2 && remainingCoupons > 0)
                {
                    maxTastinessValue = Math.Max(maxTastinessValue, Dfs(currentItemIndex + 1, remainingAmount - itemPrices[currentItemIndex] / 2, remainingCoupons - 1) + itemTastinessValues[currentItemIndex]);
                }

                // Store the result in the memoization array before returning
                memo[currentItemIndex][remainingAmount][remainingCoupons] = maxTastinessValue;
                return maxTastinessValue;
            }
        }


        /*
        2361. Minimum Costs Using the Train Line
        https://leetcode.com/problems/minimum-costs-using-the-train-line/description/
        https://algo.monster/liteproblems/2361

        */
        public class MinimumCostsUsingATrainLineSol
        {
            /*
            Approach 1: Top-Down Dynamic Programming
Complexity Analysis
Here, N is the number of stops.
•	Time complexity: O(N)
We have N stops, and for each stop, we will make two recursive calls for the two options; thus, the total number of operations could be 2∗N, and we need to find the answer to each state to solve the original problem. For each state, the time complexity is O(1) as we are just making recursive calls and finding the minimum of two integers. Hence, the total time complexity equals O(N).
•	Space complexity: O(N)
The size of array dp is 2∗N; also, there would be some stack space; the maximum number of active stack calls would be equal to N. We also need an array to store the answer ans, but generally, the space to store the answer is not considered part of the space complexity. Thus, the total space complexity equals O(N).


            */
            public long[] TopDownDP(int[] regular, int[] express, int expressCost)
            {
                long[][] dp = new long[regular.Length][];
                for (int i = 0; i < regular.Length; i++)
                {
                    Array.Fill(dp[i], -1);
                }

                Solve(regular.Length - 1, 1, dp, regular, express, expressCost);

                long[] ans = new long[regular.Length];
                // Store cost for each stop.
                for (int i = 0; i < regular.Length; i++)
                {
                    ans[i] = dp[i][1];
                }

                return ans;
            }
            long Solve(int i, int lane, long[][] dp, int[] regular, int[] express, int expressCost)
            {
                // If all stops are covered, return 0.
                if (i < 0)
                {
                    return 0;
                }

                if (dp[i][lane] != -1)
                {
                    return dp[i][lane];
                }

                // Use the regular lane; no extra cost to switch lanes if required.
                long regularLane = regular[i] + Solve(i - 1, 1, dp, regular, express, expressCost);
                // Use express lane; add expressCost if the previously regular lane was used.
                long expressLane = (lane == 1 ? expressCost : 0) + express[i]
                                                            + Solve(i - 1, 0, dp, regular, express, expressCost);

                return dp[i][lane] = Math.Min(regularLane, expressLane);
            }

            /*
            Approach 2: Bottom-Up Dynamic Programming
            Complexity Analysis
        Here, N is the number of stops.
        •	Time complexity: O(N)
        We iterate over each stop once to find the minimum cost, and hence the total time complexity is equal to O(N).
        •	Space complexity: O(N)
        The size of array dp is 2∗N. We also need an array to store the answer ans, but generally, the space to store the answer is not considered part of the space complexity. Thus, the total space complexity is equal to O(N).

            */
            public long[] BottomUpDP(int[] regular, int[] express, int expressCost)
            {
                long[] ans = new long[regular.Length];

                long[][] dp = new long[regular.Length + 1][];
                dp[0][1] = 0;
                // Need to spend expressCost, as we start from the regular lane initially.
                dp[0][0] = expressCost;

                for (int i = 1; i < regular.Length + 1; i++)
                {
                    // Use the regular lane; no extra cost to switch to the express lane.
                    dp[i][1] = regular[i - 1] + Math.Min(dp[i - 1][1], dp[i - 1][0]);
                    // Use express lane; add extra cost if the previously regular lane was used.
                    dp[i][0] = express[i - 1] + Math.Min(expressCost + dp[i - 1][1], dp[i - 1][0]);

                    ans[i - 1] = Math.Min(dp[i][0], dp[i][1]);
                }
                return ans;
            }

            /*
            Approach 3: Space-Optimized Bottom-Up Dynamic Programming
          Complexity Analysis
Here, N is the number of stops.
•	Time complexity: O(N)
We iterate over each stop once to find the minimum cost, and hence the total time complexity is equal to O(N).
•	Space complexity: O(1)
We only need two variables here, prevRegularLane and prevExpressLane, to find the following stop answer. We also need an array to store the answer ans, but generally, the space to store the answer is not considered part of the space complexity. Thus, the total space complexity is equal to constant.
  
            */
            public long[] BottomUpDPSpaceOptimal(int[] regular, int[] express, int expressCost)
            {

                long prevRegularLane = 0;
                // Need to spend expressCost, as we start from the regular lane initially.
                long prevExpressLane = expressCost;

                long[] ans = new long[regular.Length];
                for (int i = 1; i < regular.Length + 1; i++)
                {
                    // Use the regular lane; no extra cost to switch to the express lane.
                    long regularLaneCost = regular[i - 1] + Math.Min(prevRegularLane, prevExpressLane);
                    // Use express lane; add extra cost if the previously regular lane was used.
                    long expressLaneCost = express[i - 1] + Math.Min(expressCost + prevRegularLane, prevExpressLane);

                    ans[i - 1] = Math.Min(regularLaneCost, expressLaneCost);

                    prevRegularLane = regularLaneCost;
                    prevExpressLane = expressLaneCost;
                }

                return ans;

            }

        }


        /*
        815. Bus Routes
        https://leetcode.com/problems/bus-routes/description/
        https://algo.monster/liteproblems/815
        */
        public class MinNumBusesToDestinationSol
        {
            /*
            Approach 1: Breadth-First Search (BFS) with Bus Stops as Nodes
            Complexity Analysis
Here, M is the size of routes, and K is the maximum size of routes[i].
•	Time complexity: O(M^2∗K)
To store the routes for each stop we iterate over each route and for each route, we iterate over each stop, hence this step will take O(M∗K) time. In the BFS, we iterate over each route in the queue. For each route we popped, we will iterate over its stop, and for each stop, we will iterate over the connected routes in the map adjList, hence the time required will be O(M∗K∗M) or O(M^2∗K).
•	Space complexity: O(M⋅K)
The map adjList will store the routes for each stop. There can be M⋅K number of stops in routes in the worst case (each of the M routes can have K stops), possibly with duplicates. When represented using adjList, each of the mentioned stops appears exactly once. Therefore, adjList contains an equal number of stop-route element pairs.

            */
            public int BFSWithStopsAsNodes(int[][] routes, int source, int target)
            {
                if (source == target)
                {
                    return 0;
                }

                Dictionary<int, List<int>> adjList = new Dictionary<int, List<int>>();
                // Create a map from the bus stop to all the routes that include this stop.
                for (int r = 0; r < routes.Length; r++)
                {
                    foreach (int stop in routes[r])
                    {
                        // Add all the routes that have this stop.
                        List<int> route = adjList.GetValueOrDefault(
                            stop,
                            new List<int>()
                        );
                        route.Add(r);
                        adjList.Add(stop, route);
                    }
                }

                Queue<int> q = new Queue<int>();
                HashSet<int> vis = new HashSet<int>(routes.Length);
                // Insert all the routes in the queue that have the source stop.
                foreach (int route in adjList[source])
                {
                    q.Enqueue(route);
                    vis.Add(route);
                }

                int busCount = 1;
                while (q.Count > 0)
                {
                    int size = q.Count;

                    for (int i = 0; i < size; i++)
                    {
                        int route = q.Dequeue();

                        // Iterate over the stops in the current route.
                        foreach (int stop in routes[route])
                        {
                            // Return the current count if the target is found.
                            if (stop == target)
                            {
                                return busCount;
                            }

                            // Iterate over the next possible routes from the current stop.
                            foreach (int nextRoute in adjList[stop])
                            {
                                if (!vis.Contains(nextRoute))
                                {
                                    vis.Add(nextRoute);
                                    q.Enqueue(nextRoute);
                                }
                            }
                        }
                    }
                    busCount++;
                }
                return -1;
            }

            /*
            Approach 2: Breadth-First Search (BFS) with Routes as Nodes
            Complexity Analysis
Here, M is the size of routes, and K is the maximum size of routes[i].
•	Time complexity: O(M^2∗K+M∗k∗logK)
The createGraph method will iterate over every pair of M routes and for each iterate over the K stops to check if there is a common stop, this step will take O(M^2∗K). The addStartingNodes method will iterate over all the M routes and check if the route has source in it, this step will take O(M∗K). In BFS, we iterate over each of the M routes, and for each route, we iterate over the adjacent route which could be M again, so the time it takes is O(M^2).
Sorting each routes[i] takes K∗logK time.
Thus, the time complexity is equal to O(M^2∗K+M∗K∗logK).
•	Space complexity: O(M^2+logK)
The map adjList will store the routes for each route, thus the space it takes is O(M^2). The queue q and the set visited store the routes and hence can take O(M) space.
Some extra space is used when we sort routes[i] in place. The space complexity of the sorting algorithm depends on the programming language.
o	In C++, the sort() function is implemented as a hybrid of Quick Sort, Heap Sort, and Insertion Sort, with a worse-case space complexity of O(logK).
o	In Java, Arrays.sort() is implemented using a variant of the Quick Sort algorithm which has a space complexity of O(logK).
Thus, the total space complexity is equal to O(M^2+logK).

            */
            private List<List<int>> adjacencyList = new List<List<int>>();

            // Iterate over each pair of routes and add an edge between them if there's a common stop.
            private void CreateGraph(int[][] routes)
            {
                for (int i = 0; i < routes.Length; i++)
                {
                    for (int j = i + 1; j < routes.Length; j++)
                    {
                        if (HaveCommonNode(routes[i], routes[j]))
                        {
                            adjacencyList[i].Add(j);
                            adjacencyList[j].Add(i);
                        }
                    }
                }
            }

            // Returns true if the provided routes have a common stop, false otherwise.
            private bool HaveCommonNode(int[] route1, int[] route2)
            {
                int i = 0, j = 0;
                while (i < route1.Length && j < route2.Length)
                {
                    if (route1[i] == route2[j])
                    {
                        return true;
                    }

                    if (route1[i] < route2[j])
                    {
                        i++;
                    }
                    else
                    {
                        j++;
                    }
                }
                return false;
            }

            // Add all the routes in the queue that have the source as one of the stops.
            private void AddStartingNodes(Queue<int> queue, int[][] routes, int source)
            {
                for (int i = 0; i < routes.Length; i++)
                {
                    if (IsStopExist(routes[i], source))
                    {
                        queue.Enqueue(i);
                    }
                }
            }

            // Returns true if the given stop is present in the route, false otherwise.
            private bool IsStopExist(int[] route, int stop)
            {
                for (int j = 0; j < route.Length; j++)
                {
                    if (route[j] == stop)
                    {
                        return true;
                    }
                }
                return false;
            }

            public int BFSWithRoutesAsNodes(int[][] routes, int source, int target)
            {
                if (source == target)
                {
                    return 0;
                }

                for (int i = 0; i < routes.Length; ++i)
                {
                    Array.Sort(routes[i]);
                    adjacencyList.Add(new List<int>());
                }

                CreateGraph(routes);

                Queue<int> queue = new Queue<int>();
                AddStartingNodes(queue, routes, source);

                HashSet<int> visited = new HashSet<int>(routes.Length);
                int busCount = 1;
                while (queue.Count > 0)
                {
                    int size = queue.Count;

                    while (size-- > 0)
                    {
                        int node = queue.Dequeue();

                        // Return busCount, if the stop target is present in the current route.
                        if (IsStopExist(routes[node], target))
                        {
                            return busCount;
                        }

                        // Add the adjacent routes.
                        foreach (int adjacent in adjacencyList[node])
                        {
                            if (!visited.Contains(adjacent))
                            {
                                visited.Add(adjacent);
                                queue.Enqueue(adjacent);
                            }
                        }
                    }

                    busCount++;
                }

                return -1;
            }
        }

        /*
        2355. Maximum Number of Books You Can Take	
https://leetcode.com/problems/maximum-number-of-books-you-can-take/description/
https://algo.monster/liteproblems/2355

        */
        class MaxNumBooksToTakeSol
        {
            /*
            Approach 1: Dynamic Programming
Complexity Analysis
•	Time complexity: O(n).
The algorithm iterates all the shelves once from left to right.
Inside the loop, there's a while loop that pops elements from a stack. However, each element can be pushed and popped from the stack at most once. This inner while loop doesn't lead to nested iterations, so it doesn't affect the overall time complexity.
Other operations inside the loop, such as calculating dp[i], have constant time complexity.
As a result, the dominant factor in the time complexity is the single loop that iterates through the shelves, making the overall time complexity linear, O(n).
•	Space complexity: O(n).
The algorithm uses a stack to keep track of the indices of the shelves. In the worst case when all shelves from 0 to n−1 are pushed onto the stack, it may contain all n indices.
Additionally, there is an array dp containing n elements.
Therefore, the overall space complexity is O(n).	

            */
            public long DPWithStack(int[] books)
            {
                int n = books.Length;

                Stack<int> s = new Stack<int>();
                long[] dp = new long[n];

                for (int i = 0; i < n; i++)
                {
                    // While we cannot push i, we pop from the stack
                    while (s.Count > 0 && books[s.Peek()] - s.Peek() >= books[i] - i)
                    {
                        s.Pop();
                    }

                    // Compute dp[i]
                    if (s.Count == 0)
                    {
                        dp[i] = CalculateSum(books, 0, i);
                    }
                    else
                    {
                        int j = s.Peek();
                        dp[i] = dp[j] + CalculateSum(books, j + 1, i);
                    }

                    // Push the current index onto the stack
                    s.Push(i);
                }


                // Return the maximum element in dp array
                return dp.Max();
            }

            // Helper function to calculate the sum of books in a given range [l, r]
            private long CalculateSum(int[] books, int l, int r)
            {
                long cnt = Math.Min(books[r], r - l + 1);
                return (2 * books[r] - (cnt - 1)) * cnt / 2;
            }



        }

        /*
        2865. Beautiful Towers I
        https://leetcode.com/problems/beautiful-towers-i/description/
        https://algo.monster/liteproblems/2865
        */
        class MaximumSumOfHeightsSol
        {
            /*
            Time and Space Complexity
Time Complexity
The time complexity of the provided code is O(n^2), where n is the length of the maxHeights list. This is because for each element x in maxHeights, the code iterates over the elements to its left and then to its right in separate for-loops. Each of these nested loops runs at most n - 1 times in the worst case, leading to roughly 2 * (n - 1) operations for each of the n elements, thus the quadratic time complexity.
Space Complexity
The space complexity of the provided code is O(1) as it only uses a constant amount of extra space. Variables ans, n, i, x, y, t, and j are used for computations, but no additional space that scales with the input size is used. Therefore, the space used does not depend on the input size and remains constant.

            */
            // Method to calculate the maximum sum of heights, considering that each position
            // can act as a pivot, and the sum includes the minimum height from the pivot to each side.
            public static long PeakHeights(List<int> maxHeightList)
            {
                long maxSum = 0; // This will store the maximum sum of heights we find
                int listSize = maxHeightList.Count; // The size of the provided list

                // Iterate through each element in the list to consider it as a potential pivot
                for (int i = 0; i < listSize; ++i)
                {
                    int currentHeight = maxHeightList[i]; // Height at the current pivot
                    long tempSum = currentHeight; // Initialize temp sum with current pivot's height

                    // Calculate the sum of heights to the left of the pivot
                    for (int j = i - 1; j >= 0; --j)
                    {
                        currentHeight = Math.Min(currentHeight, maxHeightList[j]); // Update to the smaller height
                        tempSum += currentHeight; // Add this height to the running total for the pivot
                    }

                    currentHeight = maxHeightList[i]; // Reset height for current pivot

                    // Calculate the sum of heights to the right of the pivot
                    for (int j = i + 1; j < listSize; ++j)
                    {
                        currentHeight = Math.Min(currentHeight, maxHeightList[j]); // Update to the smaller height
                        tempSum += currentHeight; // Add this height to the running total for the pivot
                    }

                    // Update maxSum if the sum for the current pivot is greater than the previous maximum
                    maxSum = Math.Max(maxSum, tempSum);
                }

                // Return the maximum sum of heights found
                return maxSum;
            }
        }

        /*
        2866. Beautiful Towers II
        https://leetcode.com/problems/beautiful-towers-ii/description/
        https://algo.monster/liteproblems/2866
        */
        public class MaximumSumOfHeightsIISol
        {
            /*
            Approach: DP with Stack
Time and Space Complexity
Time Complexity
The given code comprises several loops iterating separately over the list maxHeights.
1.	The first for loop is a single pass over maxHeights with an inner while loop that may pop elements from stk. Each element is pushed onto stk once and popped at most once. This suggests that the overall time for this loop is O(n).
2.	The second pass is similar to the first, but it moves in the opposite direction (reverse). It also follows a similar logic with pushing and popping from stk, and thus also has O(n) time complexity.
3.	The third loop calculates values for the list f. Each iteration does constant work unless the if condition is met. If the height is not greater than or equal to the previous one, the calculation involves a difference and a potential lookup of f[j] which is constant time. Hence, this loop also runs in O(n).
4.	The fourth loop computes the g array similarly to the f array, with the same reasoning for its time complexity. So, this loop is also O(n).
5.	The last statement computes the maximum value of a + b - c for each corresponding elements of the arrays f, g, and maxHeights with the help of the zip() function. Iterating over n elements in parallel complexity is O(n).
Combining all these, we can conclude that the total time complexity is O(n).
Space Complexity
Analyzing space complexity:
1.	Three extra arrays left, right, f, and g each of length n are used, contributing 4n to the space complexity.
2.	The stk list can possibly store up to n indexes (in the worst case where the array maxHeights is sorted in ascending order). Hence, the space used by stk could be O(n).
Add all of these together, and the space complexity totals to O(n) since the constants are dropped in Big O notation.
 
            */

            public long DPWithStack(List<int> maxHeights)
            {
                int n = maxHeights.Count; // size of the maxHeights list
                Stack<int> stack = new Stack<int>(); // stack to keep track of indices
                int[] left = new int[n]; // stores indices of the previous smaller element
                int[] right = new int[n]; // stores indices of the next smaller element
                Array.Fill(left, -1); // initialize the left array with -1
                Array.Fill(right, n); // initialize the right array with n (size of maxHeights)

                // Calculate previous smaller elements
                for (int i = 0; i < n; ++i)
                {
                    int currentHeight = maxHeights[i];
                    while (stack.Count > 0 && maxHeights[stack.Peek()] > currentHeight)
                    {
                        stack.Pop();
                    }
                    if (stack.Count > 0)
                    {
                        left[i] = stack.Peek();
                    }
                    stack.Push(i);
                }
                stack.Clear(); // clear stack for reuse

                // Calculate next smaller elements
                for (int i = n - 1; i >= 0; --i)
                {
                    int currentHeight = maxHeights[i];
                    while (stack.Count > 0 && maxHeights[stack.Peek()] >= currentHeight)
                    {
                        stack.Pop();
                    }
                    if (stack.Count > 0)
                    {
                        right[i] = stack.Peek();
                    }
                    stack.Push(i);
                }
                long[] prefixSum = new long[n]; // cumulative sum from the left
                long[] suffixSum = new long[n]; // cumulative sum from the right

                // Calculate prefix sums
                for (int i = 0; i < n; ++i)
                {
                    int currentHeight = maxHeights[i];
                    if (i > 0 && currentHeight >= maxHeights[i - 1])
                    {
                        prefixSum[i] = prefixSum[i - 1] + currentHeight;
                    }
                    else
                    {
                        int j = left[i];
                        prefixSum[i] = 1L * currentHeight * (i - j) + (j >= 0 ? prefixSum[j] : 0);
                    }
                }
                // Calculate suffix sums
                for (int i = n - 1; i >= 0; --i)
                {
                    int currentHeight = maxHeights[i];
                    if (i < n - 1 && currentHeight >= maxHeights[i + 1])
                    {
                        suffixSum[i] = suffixSum[i + 1] + currentHeight;
                    }
                    else
                    {
                        int j = right[i];
                        suffixSum[i] = 1L * currentHeight * (j - i) + (j < n ? suffixSum[j] : 0);
                    }
                }

                long maxSum = 0; // variable to store the maximum sum
                                 // Compute the maximum sum by combining prefix and suffix sums
                for (int i = 0; i < n; ++i)
                {
                    maxSum = Math.Max(maxSum, prefixSum[i] + suffixSum[i] - maxHeights[i]);
                }

                return maxSum; // return the maximum sum
            }
        }

        /*
        383. Ransom Note	
        https://leetcode.com/problems/ransom-note/description/
        */
        public class CanConstructRansomNoteSol
        {
            /*
            
Approach 0: Simulation
Complexity Analysis
We'll say m is the length of the magazine, and n is the length of the ransom note.
•	Time Complexity : O(m⋅n).
Finding the letter we need in the magazine has a cost of O(m). This is because we need to perform a linear search of the magazine. Removing the letter we need from the magazine is also O(m). This is because we need to make a new string to represent it. O(m)+O(m)=O(2⋅m)=O(m) because we drop constants in big-o analysis.
So, how many times are we performing this O(m) operation? Well, we are looping through each of the n characters in the ransom note and performing it once for each letter. This is a total of n times, and so we get n⋅O(m)=O(m⋅n).
•	Space Complexity : O(m).
Creating a new magazine with one letter less requires auxillary space the length of the magazine; O(m).

            */
            public bool Simulation(string ransomNote, string magazine)
            {
                // For each character, c, in the ransom note.
                foreach (char character in ransomNote)
                {
                    // Find the index of the first occurrence of character in the magazine.
                    int index = magazine.IndexOf(character);
                    // If there are none of character left in the String, return false.
                    if (index == -1)
                    {
                        return false;
                    }
                    // Use substring to make a new string with the characters 
                    // before "index" (but not including), and the characters 
                    // after "index". 
                    magazine = magazine.Substring(0, index) + magazine.Substring(index + 1);
                }
                // If we got this far, we can successfully build the note.
                return true;
            }

            /*
            Approach 1: Fixed Length Array
            Time and Space Complexity
            The time complexity of the function canConstruct is O(m + n), where m is the length of the ransomNote string and n is the length of the magazine string. This is because the function first counts the occurrences of each character in the magazine string, which takes O(n) time, and then iterates through each character in the ransom note, which takes O(m) time. Each character decrement and comparison is an O(1) operation, thus the total time for the loop is O(m). Combined, it leads to O(m + n) time complexity.

            The space complexity of the function is O(C), where C is the size of the character set involved in the magazine and ransom note. Given that these are likely to be letters from the English alphabet, the size of the character set C is 26. This fixed size of the character set means the space taken to store the counts is independent of the lengths of the input strings and is hence constant.


            */
            public bool FixedLengthArray(string ransomNote, string magazine)
            {
                // Array to count occurrences of each letter in the magazine.
                int[] letterCounts = new int[26];

                // Populate the letterCounts array with the count of each character in the magazine.
                for (int i = 0; i < magazine.Length; i++)
                {
                    // Increment the count of the current character.
                    letterCounts[magazine[i] - 'a']++;
                }

                // Check if the ransom note can be constructed using the letters in the magazine.
                for (int i = 0; i < ransomNote.Length; i++)
                {
                    // Decrement the count of the current character, as it is used in the ransom note.
                    if (--letterCounts[ransomNote[i] - 'a'] < 0)
                    {
                        // If any letter in the ransom note is in deficit, return false.
                        return false;
                    }
                }

                // If all letters are accounted for, return true.
                return true;
            }
            /*
            Approach 2: Two HashMaps
      Complexity Analysis
    We'll say m is the length of the magazine, and n is the length of the ransom note.
    Also, let k be the number of unique characters across both the ransom note and magazine. While this is never more than 26, we'll treat it as a variable for a more accurate complexity analysis.
    The basic HashMap operations, get(...) and put(...), are O(1) time complexity.
    •	Time Complexity : O(m).
    When m<n, we immediately return false. Therefore, the worst case occurs when m≥n.
    Creating a HashMap of counts for the magazine is O(m), as each insertion/ count update is is O(1), and is done for each of the m characters.
    Likewise, creating the HashMap of counts for the ransom note is O(n).
    We then iterate over the ransom note HashMap, which contains at most n unique values, looking up their counterparts in the magazine `HashMap. This is, therefore, at worst O(n).
    This gives us O(n)+O(n)+O(m). Now, remember how we said m≥n? This means that we can simplify it to O(m)+O(m)+O(m)=3⋅O(m)=O(m), dropping the constant of 3.
    •	Space Complexity : O(k) / O(1).
    We build two HashMaps of counts; each with up to k characters in them. This means that they take up O(k) space.
    For this problem, because k is never more than 26, which is a constant, it'd be reasonable to say that this algorithm requires O(1) space.

            */



            public bool TwoHashMaps(String ransomNote, String magazine)
            {

                // Check for obvious fail case.
                if (ransomNote.Length > magazine.Length)
                {
                    return false;
                }

                // Make the count maps.
                Dictionary<char, int> ransomNoteCounts = makeCountsMap(ransomNote);
                Dictionary<char, int> magazineCounts = makeCountsMap(magazine);

                // For each unique character, c, in the ransom note:
                foreach (char c in ransomNoteCounts.Keys)
                {
                    // Check that the count of char in the magazine is equal
                    // or higher than the count in the ransom note.
                    int countInMagazine = magazineCounts.GetValueOrDefault(c, 0);
                    int countInRansomNote = ransomNoteCounts[c];
                    if (countInMagazine < countInRansomNote)
                    {
                        return false;
                    }
                }

                // If we got this far, we can successfully build the note.
                return true;

                // Takes a String, and returns a HashMap with counts of
                // each character.
                Dictionary<char, int> makeCountsMap(String s)
                {
                    Dictionary<char, int> counts = new Dictionary<char, int>();
                    foreach (char c in s.ToCharArray())
                    {
                        int currentCount = counts.GetValueOrDefault(c, 0);
                        counts.Add(c, currentCount + 1);
                    }
                    return counts;
                }



            }
            /*
                      Approach 3: One HashMap
      Complexity Analysis
      We'll say m is the length of the magazine, and n is the length of the ransom note.
      Also, let k be the number of unique characters across both the ransom note and magazine. While this is never more than 26, we'll treat it as a variable for a more accurate complexity analysis.
      The basic HashMap operations, get(...) and put(...), are O(1) time complexity.
      •	Time Complexity : O(m).
      When m<n, we immediately return false. Therefore, the worst case occurs when m≥n.
      Creating a HashMap of counts for the magazine is O(m), as each insertion/ count update is is O(1), and is done for each of the m characters.
      We then iterate over the ransom note, performing an O(1) operation for each character in it. This has a cost of O(n).
      Becuase we know that m≥n, again this simplifies to O(m).
      •	Space Complexity : O(k) / O(1).
      Same as above.
      For this problem, because k is never more than 26, which is a constant, it'd be reasonable to say that this algorithm requires O(1) space.

                      */
            public bool OneHashMap(string ransomNote, string magazine)
            {
                // Check for obvious fail case.
                if (ransomNote.Length > magazine.Length)
                {
                    return false;
                }

                // Make a counts map for the magazine.
                Dictionary<char, int> magazineCounts = MakeCountsMap(magazine);

                // For each character in the ransom note:
                foreach (char character in ransomNote.ToCharArray())
                {
                    // Get the current count for character in the magazine.
                    int countInMagazine = magazineCounts.GetValueOrDefault(character, 0);
                    // If there are none of character left, return false.
                    if (countInMagazine == 0)
                    {
                        return false;
                    }
                    // Put the updated count for character back into magazineCounts.
                    magazineCounts[character] = countInMagazine - 1;
                }

                // If we got this far, we can successfully build the note.
                return true;
                // Takes a string, and returns a dictionary with counts of
                // each character.
                Dictionary<char, int> MakeCountsMap(string inputString)
                {
                    Dictionary<char, int> characterCounts = new Dictionary<char, int>();
                    foreach (char character in inputString.ToCharArray())
                    {
                        int currentCount = characterCounts.GetValueOrDefault(character, 0);
                        characterCounts[character] = currentCount + 1;
                    }
                    return characterCounts;
                }

            }

            /*
            Approach 4: Sorting and Stacks
            Complexity Analysis
We'll say m is the length of the magazine, and n is the length of the ransom note.
•	Time Complexity : O(mlogm).
When m<n, we immediately return false. Therefore, the worst case occurs when m≥n.
Sorting the magazine is O(mlogm). Inserting the contents into the stack is O(m), which is insignificant. This, therefore, gives us O(mlogm) for creating the magazine stack.
Likewise, creating the ransom note stack is O(nlogn).
In total, the stacks contain n+m characters. For each iteration of the loop, we are either immediately returning false, or removing at least one character from the stacks. This means that the stack processing loop has to use at most O(n+m) time.
This gives us O(mlogm)+O(nlogn)+O(n+m). Now, remembering that m≥n it simplifies down to O(mlogm)+O(mlogm)+O(m+m)=2⋅O(mlogm)+O(2⋅m)=O(mlogm).
•	Space Complexity : O(m).
The magazine stack requires O(m) space, and the ransom note stack requires O(n) space. Because m≥n, this simplifies down to O(m).

            */


            private Stack<char> SortedCharStack(String s)
            {
                char[] charArray = s.ToCharArray();
                Array.Sort(charArray);
                Stack<char> stack = new Stack<char>();
                for (int i = s.Length - 1; i >= 0; i--)
                {
                    stack.Push(charArray[i]);
                }
                return stack;
            }


            public bool SortingAndStack(String ransomNote, String magazine)
            {

                // Check for obvious fail case.
                if (ransomNote.Length > magazine.Length)
                {
                    return false;
                }

                // Reverse sort the characters of the note and magazine, and then
                // put them into stacks.
                Stack<char> magazineStack = SortedCharStack(magazine);
                Stack<char> ransomNoteStack = SortedCharStack(ransomNote);

                // And now process the stacks, while both have letters remaining.
                while (magazineStack.Count > 0 && ransomNoteStack.Count > 0)
                {
                    // If the tops are the same, pop both because we have found a match.
                    if (magazineStack.Peek().Equals(ransomNoteStack.Peek()))
                    {
                        ransomNoteStack.Pop();
                        magazineStack.Pop();
                    }
                    // If magazine's top is earlier in the alphabet, we should remove that 
                    // character of magazine as we definitely won't need that letter.
                    else if (magazineStack.Peek() < ransomNoteStack.Peek())
                    {
                        magazineStack.Pop();
                    }
                    // Otherwise, it's impossible for top of ransomNote to be in magazine.
                    else
                    {
                        return false;
                    }
                }

                // Return true iff the entire ransomNote was built.
                return ransomNoteStack.Count == 0;

            }

        }

        /*
        361. Bomb Enemy
        https://leetcode.com/problems/bomb-enemy/description/

        */
        public class MaxKilledEnemiesWithOneBombSol
        {
            const char WALL = 'W';
            const char ENEMY = 'E';
            const char EMPTY = '0';



            /*
            Approach 1: Brute-force Enumeration
            Complexity Analysis
Let W be the width of the grid and H be the hight of the grid.
•	Time Complexity: O(W⋅H⋅(W+H))
o	We run an iteration over each element in the grid. In total, the number of iterations would be W⋅H.
o	Within each iteration, we need to calculate how many enemies we will kill if we place a bomb on the given cell.
In the worst case where there is no wall in the grid, we need to check (W−1+H−1) number of cells.
o	To sum up, in the worst case where all cells are empty, the number of checks we need to perform would be W⋅H⋅(W−1+H−1).
Hence the overall time complexity of the algorithm is O(W⋅H⋅(W+H)).
•	Space Complexity: O(1)
o	The size of the variables that we used in the algorithm is constant, regardless of the input.

            */
            public int Naive(char[][] grid)
            {
                if (grid.Length == 0)
                    return 0;

                int rows = grid.Length;
                int cols = grid[0].Length;

                int maxCount = 0;

                for (int row = 0; row < rows; ++row)
                {
                    for (int col = 0; col < cols; ++col)
                    {
                        if (grid[row][col] == EMPTY)
                        {
                            int hits = this.KillEnemies(row, col, grid);
                            maxCount = Math.Max(maxCount, hits);
                        }
                    }
                }

                return maxCount;
            }

            /**
             * return the number of enemies we kill, starting from the given empty cell.
             */
            private int KillEnemies(int row, int col, char[][] grid)
            {
                int enemyCount = 0;
                // look to the left side of the cell
                for (int c = col - 1; c >= 0; --c)
                {
                    if (grid[row][c] == WALL)
                        break;
                    else if (grid[row][c] == ENEMY)
                        enemyCount += 1;
                }

                // look to the right side of the cell
                for (int c = col + 1; c < grid[0].Length; ++c)
                {
                    if (grid[row][c] == WALL)
                        break;
                    else if (grid[row][c] == ENEMY)
                        enemyCount += 1;
                }

                // look to the up side of the cell
                for (int r = row - 1; r >= 0; --r)
                {
                    if (grid[r][col] == WALL)
                        break;
                    else if (grid[r][col] == ENEMY)
                        enemyCount += 1;
                }

                // look to the down side of the cell
                for (int r = row + 1; r < grid.Length; ++r)
                {
                    if (grid[r][col] == WALL)
                        break;
                    else if (grid[r][col] == ENEMY)
                        enemyCount += 1;
                }

                return enemyCount;
            }

            /*
            Approach 2: Dynamic Programming
            Complexity Analysis
            Let W be the width of the grid and H be the hight of the grid.
            •	Time Complexity: O(W⋅H)
            o	One might argue that the time complexity should be O(W⋅H⋅(W+H)), judging from the detail that we run nested loop for each cell in grid.
            If this is the case, then the time complexity of our dynamic programming approach would be the same as the brute-force approach.
            Yet this is contradicted to the fact that by applying the dynamic programming technique we reduce the redundant calculation.
            o	To estimate overall time complexity, let us take another perspective.
            Concerning each cell in the grid, we assert that it would be visited exactly three times.
            The first visit is the case where we iterate through each cell in the grid in the outer loop.
            The second visit would occur when we need to calculate the row_hits that involves with the cell.
            And finally the third visit would occur when we calculate the value of col_hits that involves with the cell.
            o	Based on the above analysis, we can say that the overall time complexity of this dynamic programming approach is O(3⋅W⋅H)=O(W⋅H).
            •	Space Complexity: O(W)
            o	In general, with the dynamic programming approach, we gain in terms of time complexity, in trade of a lost in space complexity.
            o	In our case, we allocate some variables to hold the intermediates results, namely row_hits and col_hits[*].
            Therefore, the overall space complexity of the algorithm is O(W), where W is the number of columns in the grid

            */

            public int DP(char[][] grid)
            {
                if (grid.Length == 0)
                    return 0;

                int rows = grid.Length;
                int cols = grid[0].Length;

                int maxCount = 0, rowHits = 0;
                int[] colHits = new int[cols];

                for (int row = 0; row < rows; ++row)
                {
                    for (int col = 0; col < cols; ++col)
                    {

                        // reset the hits on the row, if necessary.
                        if (col == 0 || grid[row][col - 1] == WALL)
                        {
                            rowHits = 0;
                            for (int k = col; k < cols; ++k)
                            {
                                if (grid[row][k] == WALL)
                                    // stop the scan when we hit the wall.
                                    break;
                                else if (grid[row][k] == ENEMY)
                                    rowHits += 1;
                            }
                        }

                        // reset the hits on the column, if necessary.
                        if (row == 0 || grid[row - 1][col] == WALL)
                        {
                            colHits[col] = 0;
                            for (int k = row; k < rows; ++k)
                            {
                                if (grid[k][col] == WALL)
                                    break;
                                else if (grid[k][col] == ENEMY)
                                    colHits[col] += 1;
                            }
                        }

                        // run the calculation for the empty cell.
                        if (grid[row][col] == EMPTY)
                        {
                            maxCount = Math.Max(maxCount, rowHits + colHits[col]);
                        }
                    }
                }

                return maxCount;
            }
        }

        /*
        417. Pacific Atlantic Water Flow
        https://leetcode.com/problems/pacific-atlantic-water-flow/description/

        */
        public class PacificAtlanticSol
        {
            private static readonly int[][] DIRECTIONS = new int[][] { new int[] { 0, 1 }, new int[] { 1, 0 }, new int[] { -1, 0 }, new int[] { 0, -1 } };
            private int numberOfRows;
            private int numberOfColumns;
            private int[][] landHeights;
            /*
            Approach 1: Breadth First Search (BFS)
Complexity Analysis
•	Time complexity: O(M⋅N), where M is the number of rows and N is the number of columns.
In the worst case, such as a matrix where every value is equal, we would visit every cell twice. This is because we perform 2 traversals, and during each traversal, we visit each cell exactly once. There are M⋅N cells total, which gives us a time complexity of O(2⋅M⋅N)=O(M⋅N).
•	Space complexity: O(M⋅N), where M is the number of rows and N is the number of columns.
The extra space we use comes from our queues, and the data structure we use to keep track of what cells have been visited. Similar to the time complexity, for a given ocean, the amount of space we will use scales linearly with the number of cells. For example, in the Java implementation, to keep track of what cells have been visited, we simply used 2 matrices that have the same dimensions as the input matrix.
The same logic follows for the queues - we can't have more cells in the queue than there are cells in the matrix!

            */
            public List<List<int>> BFS(int[][] matrix)
            {
                // Check if input is empty
                if (matrix.Length == 0 || matrix[0].Length == 0)
                {
                    return new List<List<int>>();
                }

                // Save initial values to parameters
                numberOfRows = matrix.Length;
                numberOfColumns = matrix[0].Length;
                landHeights = matrix;

                // Setup each queue with cells adjacent to their respective ocean
                Queue<int[]> pacificQueue = new Queue<int[]>();
                Queue<int[]> atlanticQueue = new Queue<int[]>();
                for (int i = 0; i < numberOfRows; i++)
                {
                    pacificQueue.Enqueue(new int[] { i, 0 });
                    atlanticQueue.Enqueue(new int[] { i, numberOfColumns - 1 });
                }
                for (int i = 0; i < numberOfColumns; i++)
                {
                    pacificQueue.Enqueue(new int[] { 0, i });
                    atlanticQueue.Enqueue(new int[] { numberOfRows - 1, i });
                }

                // Perform a BFS for each ocean to find all cells accessible by each ocean
                bool[][] pacificReachable = PerformBFS(pacificQueue);
                bool[][] atlanticReachable = PerformBFS(atlanticQueue);

                // Find all cells that can reach both oceans
                List<List<int>> commonCells = new List<List<int>>();
                for (int i = 0; i < numberOfRows; i++)
                {
                    for (int j = 0; j < numberOfColumns; j++)
                    {
                        if (pacificReachable[i][j] && atlanticReachable[i][j])
                        {
                            commonCells.Add(new List<int> { i, j });
                        }
                    }
                }
                return commonCells;
            }

            private bool[][] PerformBFS(Queue<int[]> queue)
            {
                bool[][] reachable = new bool[numberOfRows][];
                for (int i = 0; i < numberOfRows; i++)
                {
                    reachable[i] = new bool[numberOfColumns];
                }

                while (queue.Count > 0)
                {
                    int[] cell = queue.Dequeue();
                    // This cell is reachable, so mark it
                    reachable[cell[0]][cell[1]] = true;
                    foreach (int[] direction in DIRECTIONS) // Check all 4 directions
                    {
                        int newRow = cell[0] + direction[0];
                        int newCol = cell[1] + direction[1];
                        // Check if new cell is within bounds
                        if (newRow < 0 || newRow >= numberOfRows || newCol < 0 || newCol >= numberOfColumns)
                        {
                            continue;
                        }
                        // Check that the new cell hasn't already been visited
                        if (reachable[newRow][newCol])
                        {
                            continue;
                        }
                        // Check that the new cell has a higher or equal height,
                        // So that water can flow from the new cell to the old cell
                        if (landHeights[newRow][newCol] < landHeights[cell[0]][cell[1]])
                        {
                            continue;
                        }
                        // If we've gotten this far, that means the new cell is reachable
                        queue.Enqueue(new int[] { newRow, newCol });
                    }
                }
                return reachable;
            }

            /*
Approach 2: Depth First Search (DFS)
Complexity Analysis
•	Time complexity: O(M⋅N), where M is the number of rows and N is the number of columns.
Similar to approach 1. The dfs function runs exactly once for each cell accessible from an ocean.
•	Space complexity: O(M⋅N), where M is the number of rows and N is the number of columns.
Similar to approach 1. Space that was used by our queues is now occupied by dfs calls on the recursion stack.

            */
            public IList<IList<int>> DFS(int[][] matrix)
            {
                // Check if input is empty
                if (matrix.Length == 0 || matrix[0].Length == 0)
                {
                    return new List<IList<int>>();
                }

                // Save initial values to parameters
                numberOfRows = matrix.Length;
                numberOfColumns = matrix[0].Length;
                landHeights = matrix;
                bool[][] pacificReachable = new bool[numberOfRows][];
                bool[][] atlanticReachable = new bool[numberOfRows][];
                for (int i = 0; i < numberOfRows; i++)
                {
                    pacificReachable[i] = new bool[numberOfColumns];
                    atlanticReachable[i] = new bool[numberOfColumns];
                }

                // Loop through each cell adjacent to the oceans and start a DFS
                for (int i = 0; i < numberOfRows; i++)
                {
                    DFSRec(i, 0, pacificReachable);
                    DFSRec(i, numberOfColumns - 1, atlanticReachable);
                }
                for (int i = 0; i < numberOfColumns; i++)
                {
                    DFSRec(0, i, pacificReachable);
                    DFSRec(numberOfRows - 1, i, atlanticReachable);
                }

                // Find all cells that can reach both oceans
                IList<IList<int>> commonCells = new List<IList<int>>();
                for (int i = 0; i < numberOfRows; i++)
                {
                    for (int j = 0; j < numberOfColumns; j++)
                    {
                        if (pacificReachable[i][j] && atlanticReachable[i][j])
                        {
                            commonCells.Add(new List<int> { i, j });
                        }
                    }
                }
                return commonCells;
            }

            private void DFSRec(int row, int col, bool[][] reachable)
            {
                // This cell is reachable, so mark it
                reachable[row][col] = true;
                foreach (int[] direction in DIRECTIONS) // Check all 4 directions
                {
                    int newRow = row + direction[0];
                    int newCol = col + direction[1];
                    // Check if new cell is within bounds
                    if (newRow < 0 || newRow >= numberOfRows || newCol < 0 || newCol >= numberOfColumns)
                    {
                        continue;
                    }
                    // Check that the new cell hasn't already been visited
                    if (reachable[newRow][newCol])
                    {
                        continue;
                    }
                    // Check that the new cell has a higher or equal height,
                    // So that water can flow from the new cell to the old cell
                    if (landHeights[newRow][newCol] < landHeights[row][col])
                    {
                        continue;
                    }
                    // If we've gotten this far, that means the new cell is reachable
                    DFSRec(newRow, newCol, reachable);
                }
            }

        }

        /*
        458. Poor Pigs
        https://leetcode.com/problems/poor-pigs/description/
        */
        public class MinPigsNeededSol
        {

            public int Maths(int buckets, int minutesToDie, int minutesToTest)
            {

                // Calculate the 'base' which is the number of states a pig can be in. It's a reflection of
                // how many times you can test each pig within the total testing time 'minutesToTest'.
                int states = minutesToTest / minutesToDie + 1;

                // Initialize a counter for the number of pigs needed.
                int numberOfPigs = 0;

                //Solution 11. With Loop
                //T:O(n) : S:(1)
                // Loop to calculate the number of pigs needed. The idea is to multiply the base by itself 
                // until we reach or exceed the number of buckets.
                for (int currentBuckets = 1; currentBuckets < buckets; currentBuckets *= states)
                {
                    // Each time we are able to cover more buckets with current number of pigs, we increase the pig count.
                    numberOfPigs++;
                }

                // Return the number of pigs calculated as the result.

                //Solution 2. Using a MATH Fomuala
                //T:O(1) : S:O(1)
                // We use a small tolerance value 1e-10 in the floating-point calculation
                numberOfPigs = (int)Math.Ceiling(Math.Log(buckets) / Math.Log(states) - 1e-10);

                return numberOfPigs;
            }
        }

        /*
        733. Flood Fill
https://leetcode.com/problems/flood-fill/description/

        */
        public class FloodFillSol
        {
            /*
            Approach #1: Depth-First Search
        Complexity Analysis
•	Time Complexity: O(N), where N is the number of pixels in the image. We might process every pixel.
•	Space Complexity: O(N), the size of the implicit call stack when calling dfs.
    
            */
            public int[][] DFS(int[][] image, int sr, int sc, int newColor)
            {
                int color = image[sr][sc];
                if (color != newColor)
                {
                    DFSRec(image, sr, sc, color, newColor);
                }
                return image;
            }
            public void DFSRec(int[][] image, int r, int c, int color, int newColor)
            {
                if (image[r][c] == color)
                {
                    image[r][c] = newColor;
                    if (r >= 1)
                    {
                        DFSRec(image, r - 1, c, color, newColor);
                    }
                    if (c >= 1)
                    {
                        DFSRec(image, r, c - 1, color, newColor);
                    }
                    if (r + 1 < image.Length)
                    {
                        DFSRec(image, r + 1, c, color, newColor);
                    }
                    if (c + 1 < image[0].Length)
                    {
                        DFSRec(image, r, c + 1, color, newColor);
                    }
                }
            }
        }

        /*
        482. License Key Formatting
       https://leetcode.com/problems/license-key-formatting/description/
        */
        public class LicenseKeyFormattingSol
        {
            /*
            Approach 1: Right to Left Traversal
            Complexity Analysis
Let N be the size of the input array.
•	Time Complexity: O(N)
o	We traverse on each input string's character once in reverse order which takes O(N) time.
o	At the end, we reverse the ans thus iterating on it once, which also takes O(N) time.
o	Thus, overall we take O(N) time.
•	Space Complexity: O(1)
o	We are not using any extra space other than the output string.

            */
            public string RightToLeftTraversal(string inputString, int groupSize)
            {
                int characterCount = 0;
                int stringLength = inputString.Length;
                StringBuilder formattedLicenseKey = new StringBuilder();

                for (int index = stringLength - 1; index >= 0; index--)
                {
                    if (inputString[index] != '-')
                    {
                        formattedLicenseKey.Append(char.ToUpper(inputString[index]));
                        characterCount++;
                        if (characterCount == groupSize)
                        {
                            formattedLicenseKey.Append('-');
                            characterCount = 0;
                        }
                    }
                }

                // Make sure that the last character is not a dash
                if (formattedLicenseKey.Length > 0 && formattedLicenseKey[formattedLicenseKey.Length - 1] == '-')
                {
                    formattedLicenseKey = new StringBuilder(formattedLicenseKey.ToString(0, formattedLicenseKey.Length - 1));
                }

                // Reversing the string
                formattedLicenseKey = new StringBuilder(string.Join("", formattedLicenseKey.ToString().Reverse().ToArray()));

                return formattedLicenseKey.ToString();
            }

            /*
            
Approach 2: Left to Right Traversal
Complexity Analysis
Let N be the size of the input array.
•	Time Complexity: O(N)
o	We traverse on each input string's character once to get the count of totalChars which takes O(N) time.
o	We traverse input string for the second time in order to correctly populate ans string in groups which again takes O(N) time.
o	Thus, overall we take O(N) time.
•	Space Complexity: O(1)
o	We are not using any extra space other than the output string.

            */
            public String LeftToRightTraversal(string s, int k)
            {
                int totalChars = 0;
                for (int idx = 0; idx < s.Length; idx++)
                {
                    if (s[idx] != '-')
                    {
                        totalChars++;
                    }
                }
                int sizeOfFirstGroup = (totalChars % k);
                if (sizeOfFirstGroup == 0)
                {
                    sizeOfFirstGroup = k;
                }
                StringBuilder ans = new StringBuilder();
                int i = 0;
                int count = 0;

                while (i < s.Length)
                {
                    if (count == sizeOfFirstGroup)
                    {
                        count = 0;
                        break;
                    }
                    if (s[i] != '-')
                    {
                        count++;
                        ans.Append(Char.ToUpper(s[i]));
                    }
                    i++;
                }
                /* This case will only appear if the value of k is greater than the total number 
                   of alphanumeric characters in string s */
                if (i >= s.Length)
                {
                    return ans.ToString();
                }
                ans.Append('-');
                while (i < s.Length)
                {
                    if (s[i] != '-')
                    {
                        /* Whenever the count is equal to k, we put a '-' after each group */
                        if (count == k)
                        {
                            ans.Append('-');
                            count = 0;
                        }
                        ans.Append(Char.ToUpper(s[i]));
                        count++;
                    }
                    i++;
                }
                return ans.ToString();
            }
        }


        /*
        486. Predict the Winner
        https://leetcode.com/problems/predict-the-winner/description/

        */
        public class PredictTheWinnerSol
        {
            /*
            Approach 1: Recursion
            Complexity Analysis
Let n be the length of the input array nums.
•	Time complexity: O(2^n)
o	The depth of the call stack will be n once the base cases are reached because each call moves left and right closer by one.
o	For each problem except the base case, the current player can either pick an element from the front or end. That means each call to maxDiff creates two more calls to maxDiff.
Thus the recursion tree having a depth of n has O(2^n) nodes, which is the time it takes to finish the search. Note that the constraints have n <= 20. This approach would not be feasible for larger values of n.

            */
            public bool Rec(int[] nums)
            {
                int n = nums.Length;

                return MaxDiff(nums, 0, n - 1) >= 0;
            }
            private int MaxDiff(int[] nums, int left, int right)
            {
                if (left == right)
                {
                    return nums[left];
                }

                int scoreByLeft = nums[left] - MaxDiff(nums, left + 1, right);
                int scoreByRight = nums[right] - MaxDiff(nums, left, right - 1);

                return Math.Max(scoreByLeft, scoreByRight);
            }
            /*
            Approach 2: Dynamic Programming, Top-Down
            Complexity Analysis
Let n be the length of the input array nums.
•	Time complexity: O(n2)
o	We use a cache memo to store the computed states. During the recursion, the cache makes sure we don't calculate a state more than once. The number of states (left, right) is O(n2).
•	Space complexity: O(n2)
o	We use a hashmap or a two-dimensional array memo as memory, it contains at most O(n2) distinct states.

            */
            int[][] memo;

            public bool TopDownDP(int[] nums)
            {
                int n = nums.Length;
                memo = new int[n][];
                for (int i = 0; i < n; ++i)
                {
                    Array.Fill(memo[i], -1);
                }

                return MaxDiff(nums, 0, n - 1) >= 0;
                int MaxDiff(int[] nums, int left, int right)
                {
                    if (memo[left][right] != -1)
                    {
                        return memo[left][right];
                    }
                    if (left == right)
                    {
                        return nums[left];
                    }

                    int scoreByLeft = nums[left] - MaxDiff(nums, left + 1, right);
                    int scoreByRight = nums[right] - MaxDiff(nums, left, right - 1);
                    memo[left][right] = Math.Max(scoreByLeft, scoreByRight);

                    return memo[left][right];
                }
            }

            /*
            Approach 3: Dynamic Programming, Bottom-Up
            Complexity Analysis
            Let n be the length of the input array nums.
            •	Time complexity: O(n2)
            o	We create a 2D array of size n×n as memory. The value of each cell is computed once, which takes O(1) time.
            •	Space complexity: O(n2)
            o	We create a 2D array of size n×n as memory.

            */
            public bool BottomUpDP(int[] nums)
            {
                int n = nums.Length;
                int[][] dp = new int[n][];
                for (int i = 0; i < n; ++i)
                {
                    dp[i][i] = nums[i];
                }

                for (int diff = 1; diff < n; ++diff)
                {
                    for (int left = 0; left < n - diff; ++left)
                    {
                        int right = left + diff;
                        dp[left][right] = Math.Max(nums[left] - dp[left + 1][right],
                                                  nums[right] - dp[left][right - 1]);
                    }
                }

                return dp[0][n - 1] >= 0;
            }

            /*
            Approach 4: Dynamic Programming, Space-Optimized
            Complexity Analysis
            Let n be the length of the input array nums.
            •	Time complexity: O(n2)
            o	We fill n cells of dp in the 1st round, then n - 1 cells in the 2nd round, and so on. The total number of updated cells is O(n2), each cell takes constant time.
            •	Space complexity: O(n)
            o	We only use an array of size n.

            */
            public bool predictTheWinner(int[] nums)
            {
                int n = nums.Length;
                int[] dp = new int[n];

                Array.Copy(nums, dp, n);

                for (int diff = 1; diff < n; ++diff)
                {
                    for (int left = 0; left < n - diff; ++left)
                    {
                        int right = left + diff;
                        dp[left] = Math.Max(nums[left] - dp[left + 1], nums[right] - dp[left]);
                    }
                }

                return dp[0] >= 0;
            }


        }



        /*
        502. IPO
        https://leetcode.com/problems/ipo/description/
        */
        class FindMaximizedCapitalSol
        {

            /*            
            A Greedy Approach With Sort & Priority Queue
    Complexity Analysis
Let n be the number of projects.
•	Time complexity: O(nlogn). Sorting the projects by increasing capital takes O(nlogn) time. Also, we perform O(n) operations with the priority queue, each in O(logn).
•	Space complexity: O(n). The sorted array of projects and the priority queue take linear space.
        
            */
            public int GreedyWithSortAndPQ(int k, int w, int[] profits, int[] capital)
            {
                int n = profits.Length;
                Project[] projects = new Project[n];
                for (int i = 0; i < n; i++)
                {
                    projects[i] = new Project(capital[i], profits[i]);
                }
                Array.Sort(projects);
                // PriorityQueue is a min heap, but we need a max heap, so we use
                // Negation as in -10 to keep min heap disguising as max heap.
                PriorityQueue<int, int> q = new PriorityQueue<int, int>();
                int ptr = 0;
                for (int i = 0; i < k; i++)
                {
                    while (ptr < n && projects[ptr].Capital <= w)
                    {
                        var profit = projects[ptr++].Profit;
                        q.Enqueue(profit, -profit);
                    }
                    if (q.Count == 0)
                    {
                        break;
                    }
                    w += q.Dequeue();
                }
                return w;
            }
            class Project : IComparable<Project>
            {
                public int Capital, Profit;

                public Project(int capital, int profit)
                {
                    this.Capital = capital;
                    this.Profit = profit;
                }

                public int CompareTo(Project project)
                {
                    return Capital - project.Capital;
                }
            }


        }

        /*
        495. Teemo Attacking
        https://leetcode.com/problems/teemo-attacking/description/
        */
        public class FindPoisonedDurationSol
        {
            /*           
Approach 1: One pass
Complexity Analysis
•	Time complexity: O(N), where N is the length of the input list since we iterate the entire list.
•	Space complexity: O(1), it's a constant space solution.

            */
            public int OnePass(int[] timeSeries, int duration)
            {
                int n = timeSeries.Length;
                if (n == 0) return 0;

                int total = 0;
                for (int i = 0; i < n - 1; ++i)
                    total += Math.Min(timeSeries[i + 1] - timeSeries[i], duration);
                return total + duration;
            }
        }

        /*
        853. Car Fleet
      https://leetcode.com/problems/car-fleet/description/
      https://algo.monster/liteproblems/853
        */
        public class CarFleetSol
        {
            /*
            Approach1: 1DArray and Sorting
        O(NlogN) Quick sort the cars by position. (Other sort can be applied)
O(N) One pass for all cars from the end to start (another direction also works).
    
            */

            // Function to count the number of car fleets that will arrive at the target
            public int With1DArrayAndSorting(int target, int[] positions, int[] speeds)
            {
                // Number of cars
                int carCount = positions.Length;
                // Array to hold the indices of the cars
                int[] indices = new int[carCount];

                // Populate the indices array with the array indices
                for (int i = 0; i < carCount; ++i)
                {
                    indices[i] = i;
                }

                // Sort the indices based on the positions of the cars in descending order
                Array.Sort(indices, (a, b) => positions[b] - positions[a]);

                // Count of car fleets
                int fleetCount = 0;
                // The time taken by the previous car to reach the target
                double previousTime = 0;

                // Iterate through the sorted indices array
                foreach (int index in indices)
                {
                    // Calculate the time taken for the current car to reach the target
                    double timeToReach = 1.0 * (target - positions[index]) / speeds[index];

                    // If the time taken is greater than the previous time, it forms a new fleet
                    if (timeToReach > previousTime)
                    {
                        fleetCount++;
                        previousTime = timeToReach; // Update the previous time
                    }
                    // If the time is less or equal, it joins the fleet of the previous car
                }
                // Return the total number of fleets
                return fleetCount;
            }

            /*
            Approach2: 2DArray with Sorting
        O(NlogN) Quick sort the cars by position. (Other sort can be applied)
O(N) One pass for all cars from the end to start (another direction also works).
    
            */


            public int With2DArrayAndSorting(int target, int[] pos, int[] speed)
            {
                int N = pos.Length, res = 0;
                double[][] cars = new double[N][];
                for (int i = 0; i < N; ++i)
                    cars[i] = new double[] { pos[i], (double)(target - pos[i]) / speed[i] };
                Array.Sort(cars, (a, b) => a[0].CompareTo(b[0]));
                double cur = 0;
                for (int i = N - 1; i >= 0; --i)
                {
                    if (cars[i][1] > cur)
                    {
                        cur = cars[i][1];
                        res++;
                    }
                }
                return res;
            }
            /*
            Approach3: with Sorted Dictionary
        O(NlogN) Quick sort the cars by position. (Other sort can be applied)
O(N) One pass for all cars from the end to start (another direction also works).
    
            */

            public int WithSortedDict(int target, int[] positions, int[] speeds)
            {
                SortedDictionary<int, double> sortedDictionary = new SortedDictionary<int, double>(Comparer<int>.Create((x, y) => y.CompareTo(x)));
                for (int i = 0; i < positions.Length; ++i)
                    sortedDictionary[positions[i]] = (double)(target - positions[i]) / speeds[i];

                int resultCount = 0;
                double currentMaxTime = 0;

                foreach (double time in sortedDictionary.Values)
                {
                    if (time > currentMaxTime)
                    {
                        currentMaxTime = time;
                        resultCount++;
                    }
                }

                return resultCount;

            }


        }

        /*
        1776. Car Fleet II
        https://leetcode.com/problems/car-fleet-ii/description/

        */
        public class GetCollisionTimesSol
        {
            /*
            Complexity
    Time O(n)
    Space O(n)

            */
            public double[] GetCollisionTimes(int[][] cars)
            {
                int n = cars.Length;
                LinkedList<int> stack = new LinkedList<int>();

                double[] res = new double[n];
                for (int i = n - 1; i >= 0; --i)
                {
                    res[i] = -1.0;
                    int p = cars[i][0], s = cars[i][1];
                    while (stack.Count > 0)
                    {
                        int j = stack.Last.Value, p2 = cars[j][0], s2 = cars[j][1];
                        if (s <= s2 || 1.0 * (p2 - p) / (s - s2) >= res[j] && res[j] > 0)
                            stack.RemoveLast();
                        else
                            break;
                    }
                    if (stack.Count > 0)
                    {
                        int j = stack.Last(), p2 = cars[j][0], s2 = cars[j][1];
                        res[i] = 1.0 * (p2 - p) / (s - s2);
                    }
                    stack.AddLast(i);
                }
                return res;

            }
        }
        /*
        2751. Robot Collisions	
        https://leetcode.com/problems/robot-collisions/description/

        */
        public class SurvivedRobotsHealthsSol
        {

            /*
            Approach: Sorting & Stack
Complexity Analysis
Let n be the number of robots.
•	Time Complexity: O(n⋅logn)
Sorting the robots based on their positions takes O(nlogn) time.
Initializing the indices array takes O(n) time.
The for loop that processes each robot runs in O(n) time since each robot is processed once.
Therefore, the overall time complexity is dominated by the sorting step, making it O(n⋅logn).
•	Space Complexity: O(n)
In Python, the sort method uses Timsort, which has a worst-case space complexity of O(n) due to the additional space used by the merge operations.
In Java, Arrays.sort() uses a variant of Quick Sort for primitive types, with a space complexity of O(logn).
In C++, the sort() function typically uses a hybrid of Quick Sort, Heap Sort, and Insertion Sort, with a worst-case space complexity of O(logn).
Apart from the sorting step, we use an additional space of O(n) for the indices array.
The stack in the worst case holds O(n) elements.
Therefore, the total space complexity is O(n).

            */
            public List<int> SurvivedRobotsHealths(
                int[] positions,
                int[] healths,
                String directions
            )
            {
                int n = positions.Length;
                int[] indices = new int[n];
                List<int> result = new List<int>();
                Stack<int> stack = new Stack<int>();

                for (int index = 0; index < n; ++index)
                {
                    indices[index] = index;
                }

                Array.Sort(
                    indices,
                    (lhs, rhs) => positions[lhs].CompareTo(positions[rhs])
                );

                foreach (int currentIndex in indices)
                {
                    // Add right-moving robots to the stack
                    if (directions[currentIndex] == 'R')
                    {
                        stack.Push(currentIndex);
                    }
                    else
                    {
                        while (stack.Count > 0 && healths[currentIndex] > 0)
                        {
                            // Pop the top robot from the stack for collision check
                            int topIndex = stack.Pop();

                            // Top robot survives, current robot is destroyed
                            if (healths[topIndex] > healths[currentIndex])
                            {
                                healths[topIndex] -= 1;
                                healths[currentIndex] = 0;
                                stack.Push(topIndex);
                            }
                            else if (healths[topIndex] < healths[currentIndex])
                            {
                                // Current robot survives, top robot is destroyed
                                healths[currentIndex] -= 1;
                                healths[topIndex] = 0;
                            }
                            else
                            {
                                // Both robots are destroyed
                                healths[currentIndex] = 0;
                                healths[topIndex] = 0;
                            }
                        }
                    }
                }

                // Collect surviving robots
                for (int index = 0; index < n; ++index)
                {
                    if (healths[index] > 0)
                    {
                        result.Add(healths[index]);
                    }
                }
                return result;
            }
        }


        /*
        1503. Last Moment Before All Ants Fall Out of a Plank
        https://leetcode.com/problems/last-moment-before-all-ants-fall-out-of-a-plank/description/
        */
        class GetLastMomentSol
        {
            /*
            Approach: Ants Pass Each Other!
            Complexity Analysis
Given n as the length of left and m as the length of right,
•	Time complexity: O(n+m)
We iterate over left and right once, performing O(1) work at each iteration.
•	Space complexity: O(1)
We aren't using any extra space except the for loop iteration variable.

            */
            public int PassingEachOther(int n, int[] left, int[] right)
            {
                int ans = 0;
                foreach (int num in left)
                {
                    ans = Math.Max(ans, num);
                }

                foreach (int num in right)
                {
                    ans = Math.Max(ans, n - num);
                }

                return ans;
            }
        }


        /*

    649. Dota2 Senate
    https://leetcode.com/problems/dota2-senate/description/

        */

        public class PredictPartyVictorySol
        {
            /* 
            Approach 1: Greedy
Complexity Analysis
Let N be the number of senators in the senate.
•	Time complexity: O(N^2).
o	Counting the number of senators of each type is O(N) time.
o	As discussed in Overview, there will be O(N) turns/votes.
Each turn will take O(N) time to find the next senator to ban. Also, removing an element from an array is O(N) time. Thus, each turn requires O(2N) operations, which is O(N) time.
Thus, O(N) turns/votes requires O(N^2) time.
Hence, the overall time complexity will be O(N+N^2)=O(N^2).
•	Space complexity: O(N).
If the string is mutable, then we can do it in place.
However, strings are often immutable. Thus, we need to use a new data structure of size N to store the senate. Hence, the space complexity will be O(N).


            */
            public string Greedy(string senate)
            {

                // Count of Each Type of Senator to check for Winner
                int rCount = senate.Count(c => c == 'R');
                int dCount = senate.Length - rCount;

                // Ban the candidate "toBan", immediate next to "startAt"
                // If have to loop around, then it means next turn will be of
                // senator at same index. Returns loop around boolean
                bool ban(char toBan, int startAt)
                {
                    bool loopAround = false;
                    int pointer = startAt;

                    while (true)
                    {
                        if (pointer == 0)
                        {
                            loopAround = true;
                        }
                        if (senate[pointer] == toBan)
                        {
                            senate = senate.Remove(pointer, 1);
                            break;
                        }
                        pointer = (pointer + 1) % senate.Length;
                    }

                    return loopAround;
                }

                // Turn of Senator at this index
                int turn = 0;

                // While No Winner
                while (rCount > 0 && dCount > 0)
                {

                    // Ban the next opponent, starting at one index ahead
                    // Taking MOD to loop around.
                    // If index of banned senator is before current index,
                    // then we need to decrement turn by 1, as we have removed
                    // a senator from list
                    if (senate[turn] == 'R')
                    {
                        bool bannedSenatorBefore = ban('D', (turn + 1) % senate.Length);
                        dCount--;
                        if (bannedSenatorBefore)
                        {
                            turn--;
                        }
                    }
                    else
                    {
                        bool bannedSenatorBefore = ban('R', (turn + 1) % senate.Length);
                        rCount--;
                        if (bannedSenatorBefore)
                        {
                            turn--;
                        }
                    }

                    // Increment turn by 1
                    turn = (turn + 1) % senate.Length;
                }

                // Return Winner depending on count
                return dCount == 0 ? "Radiant" : "Dire";
            }

            /*
            Approach 2: Boolean Array
Complexity Analysis
Let N be the number of senators in the senate.
•	Time complexity: O(N^2).
o	Counting the number of senators of each type is O(N) time.
o	As discussed in Overview, there will be at most N turns. Thus, if !banned[turn] in while (rCount > 0 && dCount > 0) will be executed at most N times.
In each turn, we will iterate over the entire senate string to find the next eligible senator to ban. This is bounded by N as well.
Thus, the overall time complexity is O(N^2).
•	Space complexity: O(N).
We use a boolean array of size N to mark banned senators. However, compared to previous approach, we have overcome the nuances of maintaining the turn invariant.

            */
            public string BoolArray(string senate)
            {

                // Count of Each Type of Senator to check for Winner
                int rCount = senate.Count(x => x == 'R');
                int dCount = senate.Length - rCount;

                // To mark Banned Senators
                bool[] banned = new bool[senate.Length];

                // Ban the candidate "toBan", immediate next to "startAt"
                Action<char, int> ban = (toBan, startAt) =>
                {
                    // Find the next eligible senator of "toBan" type
                    // On found, mark him as banned
                    while (true)
                    {
                        if (senate[startAt] == toBan && !banned[startAt])
                        {
                            banned[startAt] = true;
                            break;
                        }
                        startAt = (startAt + 1) % senate.Length;
                    }
                };

                // Turn of Senator at this Index
                int turn = 0;

                // While both parties have at least one senator
                while (rCount > 0 && dCount > 0)
                {

                    if (!banned[turn])
                    {
                        if (senate[turn] == 'R')
                        {
                            ban('D', (turn + 1) % senate.Length);
                            dCount--;
                        }
                        else
                        {
                            ban('R', (turn + 1) % senate.Length);
                            rCount--;
                        }
                    }

                    turn = (turn + 1) % senate.Length;
                }

                // Return Winner
                return dCount == 0 ? "Radiant" : "Dire";
            }



            /*
            Approach 3: Binary Search
Complexity Analysis
Let N be the number of senators in the senate.
•	Time complexity: O(N^2).
o	Creating the list of indices of eligible senators takes O(N) time.
o	The if !banned[turn] condition in the while (!rIndices.empty() && !dIndices.empty()) loop is executed N times. Because there will be at most O(N) vote as discussed in Overview.
Now, each vote will call the ban function. The ban function uses Binary Search to find the index of the senator to ban. The Binary Search takes O(logN) time. But, it is also removing the index from the list using the erase (or equivalent) function. This takes O(N) time. So, the total time taken by the ban function is O(N).
Hence, the total time taken by the while loop is O(N^2).
•	Thus, the total time complexity is O(N2).
•	Side Note : If popping to maintain invariant of eligible senators was O(1), then the time complexity would have been O(N+NlogN)=O(NlogN).
•	Space complexity: O(N).
o	The space taken by the banned array is O(N).
o	The space taken by the rIndices and dIndices array is O(N).
o	Thus, the total space complexity is O(N).

            */

            public string BinarySearch(string senate)
            {

                // Number of Senators
                int n = senate.Length;

                // To mark Banned Senators
                bool[] banned = new bool[n];

                // List of indices of Eligible Radiant and Dire Senators
                List<int> rIndices = new List<int>();
                List<int> dIndices = new List<int>();
                for (int i = 0; i < n; i++)
                {
                    if (senate[i] == 'R')
                        rIndices.Add(i);
                    else
                        dIndices.Add(i);
                }

                // Ban the senator of "indices" immediate next to "startAt"
                Action<List<int>, int> ban = (indices, startAt) =>
                {
                    // Find the index of "index of senator to ban" using Binary Search
                    int temp = indices.BinarySearch(startAt);

                    // If startAt is more than the last index,
                    // then start from the beginning. Ban the first senator
                    if (temp < 0)
                    {
                        temp = ~temp;
                        if (temp == indices.Count)
                        {
                            banned[indices[0]] = true;
                            indices.RemoveAt(0);
                        }

                        // Else, Ban the senator at the index
                        else
                        {
                            banned[indices[temp]] = true;
                            indices.RemoveAt(temp);
                        }
                    }

                    // Else, Ban the senator at the index
                    else
                    {
                        banned[indices[temp]] = true;
                        indices.RemoveAt(temp);
                    }
                };

                // Turn of Senator at this Index
                int turn = 0;

                // While both parties have at least one senator
                while (rIndices.Count > 0 && dIndices.Count > 0)
                {

                    if (!banned[turn])
                    {
                        if (senate[turn] == 'R')
                            ban(dIndices, turn);
                        else
                            ban(rIndices, turn);
                    }

                    turn = (turn + 1) % n;
                }

                // Return the party with at least one senator
                return dIndices.Count == 0 ? "Radiant" : "Dire";
            }


            /*        
    Approach 4: Two Queues
Complexity Analysis
Let N be the number of senators in the senate.
•	Time complexity: O(N).
o	Populating the queues takes O(N) time.
o	While loop will give chance to each eligible senator to vote until the last round. The voting process for one senator takes O(1) time because of constant queue operations. There will be O(N) such votes as discussed in Overview section.
o	Hence, total time complexity is O(N+N)=O(N).
•	Space complexity: O(N).
Storing the index of senators in the queues takes O(N) space. The queues will either decrease or remain the same in size in each round. They will never increase in size. Hence, space complexity is O(N).

            */
            public string TwoQueues(string senate)
            {

                // Number of Senator
                int n = senate.Length;

                // Queues with Senator's Index.
                // Index will be used to find the next turn of Senator
                Queue<int> rQueue = new Queue<int>();
                Queue<int> dQueue = new Queue<int>();

                // Populate the Queues
                for (int i = 0; i < n; i++)
                {
                    if (senate[i] == 'R')
                    {
                        rQueue.Enqueue(i);
                    }
                    else
                    {
                        dQueue.Enqueue(i);
                    }
                }

                // While both parties have at least one Senator
                while (rQueue.Count > 0 && dQueue.Count > 0)
                {

                    // Pop the Next-Turn Senate from both Q.
                    int rTurn = rQueue.Dequeue();
                    int dTurn = dQueue.Dequeue();

                    // ONE having a larger index will be banned by a lower index
                    // Lower index will again get Turn, so EN-Queue again
                    // But ensure its turn comes in the next round only
                    if (dTurn < rTurn)
                    {
                        dQueue.Enqueue(dTurn + n);
                    }
                    else
                    {
                        rQueue.Enqueue(rTurn + n);
                    }
                }

                // One's which Empty is not winner
                return rQueue.Count == 0 ? "Dire" : "Radiant";
            }
            /*

    Approach 5: Single Queue
Let N be the number of senators in the senate.
•	Time complexity: O(N).
o	Counting the number of senators of each party takes O(N) time. So does populating the queue.
o	The condition while (rCount && dCount) will be executed O(N) times because they are the simulation of the voting process, which is bounded by O(N) as discussed in Overview section.
Inside the loop, there are O(1) operations.
o	So the total time complexity is O(N+N)=O(N).
•	Space complexity: O(N).
The Queue will have N senators initially. The number can only decrease but can never increase. So the space complexity is O(N).

            */
            public string SingleQueue(string senate)
            {

                // Number of Senators of each party
                int rCount = 0, dCount = 0;

                // Floating Ban Count
                int dFloatingBan = 0, rFloatingBan = 0;

                // Queue of Senators
                Queue<char> q = new Queue<char>();
                foreach (char c in senate)
                {
                    q.Enqueue(c);
                    if (c == 'R') rCount++;
                    else dCount++;
                }

                // While any party has eligible Senators
                while (rCount > 0 && dCount > 0)
                {

                    // Pop the senator with turn
                    char curr = q.Dequeue();

                    // If eligible, float the ban on the other party, enqueue again.
                    // If not, decrement the floating ban and count of the party.
                    if (curr == 'D')
                    {
                        if (dFloatingBan > 0)
                        {
                            dFloatingBan--;
                            dCount--;
                        }
                        else
                        {
                            rFloatingBan++;
                            q.Enqueue('D');
                        }
                    }
                    else
                    {
                        if (rFloatingBan > 0)
                        {
                            rFloatingBan--;
                            rCount--;
                        }
                        else
                        {
                            dFloatingBan++;
                            q.Enqueue('R');
                        }
                    }
                }

                // Return the party with eligible Senators
                return rCount > 0 ? "Radiant" : "Dire";
            }


        }


        /* 398. Random Pick Index
        https://leetcode.com/problems/random-pick-index/description/?envType=company&envId=facebook&favoriteSlug=facebook-all&difficulty=MEDIUM
        https://algo.monster/liteproblems/398
         */
        class RandomPickIndexSol
        {
            private int[] nums; // This array holds the original array of numbers.
            private Random random = new Random(); // Random object to generate random numbers.

            // Constructor that receives an array of numbers.
            public RandomPickIndexSol(int[] nums)
            {
                this.nums = nums; // Initialize the nums array with the given input array.
            }

            /* Time and Space Complexity
        Time Complexity
        The time complexity of the pick method is O(N), where N is the total number of elements in nums. This is because, in the worst case, we have to iterate over all the elements in nums to find all occurrences of target and decide whether to pick each occurrence or not.

        During each iteration, we perform the following operations:

        Compare the current value v with target.
        If they match, we increment n.
        Generate a random number x with random.randint(1, n), which has constant time complexity O(1).
        Compare x to n and possibly update ans.
        These operations are within the single pass loop through nums, hence maintaining the overall time complexity of O(N).

        Space Complexity
        The space complexity of the pick method is O(1). The additional space required for the method execution does not depend on the size of the input array but only on a fixed set of variables (n, ans, i, and v), which use a constant amount of space.

        The class Solution itself has space complexity O(N), where N is the number of elements in nums, since it stores the entire list of numbers.

        However, when analyzing the space complexity of the pick method, we consider only the extra space used by the method excluding the space used to store the input, which in this case remains constant.

         */        // Method to pick a random index where the target value is found in the nums array.
            public int Pick(int target)
            {
                int count = 0; // Counter to track how many times we've seen the target so far.
                int result = 0; // Variable to keep the result index.

                // Iterating over the array to find target.
                for (int i = 0; i < nums.Length; ++i)
                {
                    if (nums[i] == target)
                    { // Check if current element is the target.
                        count++; // Increment the count since we have found the target.
                                 // Generate a random number between 1 and the number of times target has been seen inclusively.
                        int randomNumber = 1 + random.Next(count);

                        // If the random number equals to the count (probability 1/n),
                        // set the result to current index i.
                        if (randomNumber == count)
                        {
                            result = i;
                        }
                    }
                }
                return result; // Return the index of target chosen uniformly at random.
            }
        }

        /*
        528. Random Pick with Weight
    https://leetcode.com/problems/random-pick-with-weight/description/

        */
        class RandomPickWithWeightSol
        {
            private int[] prefixSums;
            private int totalSum;

            public RandomPickWithWeightSol(int[] w)
            {
                this.prefixSums = new int[w.Length];

                int prefixSum = 0;
                for (int i = 0; i < w.Length; ++i)
                {
                    prefixSum += w[i];
                    this.prefixSums[i] = prefixSum;
                }
                this.totalSum = prefixSum;
            }

            /*
            Approach 1: Prefix Sums with Linear Search
            Complexity Analysis
Let N be the length of the input list.
•	Time Complexity
o	For the constructor function, the time complexity would be O(N), which is due to the construction of the prefix sums.

o	For the pickIndex() function, its time complexity would be O(N) as well, since we did a linear search on the prefix sums.

•	Space Complexity
o	For the constructor function, the space complexity would be O(N), which is again due to the construction of the prefix sums.

            */
            public int PickIndexWithPrefixSumAndLinearSearch()
            {
                double target = this.totalSum * new Random().NextDouble();
                int i = 0;
                // run a linear search to find the target zone
                for (; i < this.prefixSums.Length; ++i)
                {
                    if (target < this.prefixSums[i])
                        return i;
                }
                // to have a return statement, though this should never happen.
                return i - 1;
            }

            /*            
Approach 2: Prefix Sums with Binary Search
Complexity Analysis
Let N be the length of the input list.
•	Time Complexity
o	For the constructor function, the time complexity would be O(N), which is due to the construction of the prefix sums.

o	For the pickIndex() function, this time its time complexity would be O(logN), since we did a binary search on the prefix sums.
•	Space Complexity
o	For the constructor function, the space complexity remains O(N), which is again due to the construction of the prefix sums.

o	For the pickIndex() function, its space complexity would be O(1), since it uses constant memory. Note, here we consider the prefix sums that it operates on, as the input of the function.


            */
            public int PickIndexWithPrefixSumAndBinarySearch()
            {
                double target = this.totalSum * new Random().NextDouble();

                // run a binary search to find the target zone
                int low = 0, high = this.prefixSums.Length;
                while (low < high)
                {
                    // better to avoid the overflow
                    int mid = low + (high - low) / 2;
                    if (target > this.prefixSums[mid])
                        low = mid + 1;
                    else
                        high = mid;
                }
                return low;
            }
        }

        /*
        551. Student Attendance Record I
        https://leetcode.com/problems/student-attendance-record-i/description/	
        */
        public class CheckRecordSol
        {
            /*
            Approach #1 Simple Solution 
            Complexity Analysis
•	Time complexity : O(n). Single loop and indexOf method takes O(n) time.
•	Space complexity : O(1). Constant space is used.

            */
            public bool SimpleSol(String s)
            {
                int count = 0;
                for (int i = 0; i < s.Length; i++)
                    if (s[i] == 'A')
                        count++;
                return count < 2 && s.IndexOf("LLL") < 0;
            }

            /*
            Approach #2 Better Solution
            Complexity Analysis
•	Time complexity : O(n). Single loop and indexOf method takes O(n) time.
•	Space complexity : O(1). Constant space is used.

            */
            public bool BetterSol(String s)
            {
                int count = 0;
                for (int i = 0; i < s.Length && count < 2; i++)
                    if (s[i] == 'A')
                        count++;
                return count < 2 && s.IndexOf("LLL") < 0;
            }
            /*
            Approach #3 Single pass Solution (Without indexOf method) 
            **Complexity Analysis**
•	Time complexity : O(n). Single loop upto string length is used.
•	Space complexity : O(1). Constant space is used.

            */
            public bool SinglePassWithoutIndexOf(String s)
            {
                int countA = 0;
                for (int i = 0; i < s.Length && countA < 2; i++)
                {
                    if (s[i] == 'A')
                        countA++;
                    if (i <= s.Length - 3 && s[i] == 'L' && s[i + 1] == 'L' && s[i + 2] == 'L')
                        return false;
                }
                return countA < 2;
            }
            /*
            Approach #4 Using Regex 
Complexity Analysis
•	Time complexity : O(n). matches method takes O(n) time.
•	Space complexity : O(1). No Extra Space is used.

            */
            public bool checkRecord(string s)
            {
                var regex = new Regex(@".*(A.*A|LLL).*");

                return !regex.IsMatch(s);
            }
        }

        /*
        552. Student Attendance Record II
    https://leetcode.com/problems/student-attendance-record-ii/description/	
        */
        class CheckRecordIISol
        {
            private const int MOD = 1000000007;
            // Cache to store sub-problem results.
            private int[][][] memo;

            /*
            Approach 1: Top-Down Dynamic Programming with Memoization
            Complexity Analysis
            •	Time complexity: O(n)
            o	Our recursive function will only evaluate n×2×3 unique sub-problems due to memoization.
            o	So, this approach will take O(6⋅n)=O(n) time.
            •	Space complexity: O(n)
            o	We initialized an additional array memo of size n×2×3 that takes O(n) space.
            o	The recursive call stack will also take O(n) space in the worst-case.
            o	So, this approach will take O(6⋅n+n)=O(n) space.

            */
            public int TopDownWithDPMemo(int n)
            {
                // Initialize the cache.
                memo = new int[n + 1][][];
                for (int i = 0; i <= n; i++)
                {
                    memo[i] = new int[2][];
                    for (int j = 0; j < 2; j++)
                    {
                        memo[i][j] = new int[3];
                        Array.Fill(memo[i][j], -1);
                    }
                }
                // Return count of combinations of length 'n' eligible for the award.
                return EligibleCombinations(n, 0, 0);
            }

            // Recursive function to return the count of combinations of length 'n' eligible for the award.
            private int EligibleCombinations(int n, int totalAbsences, int consecutiveLates)
            {
                // If the combination has become not eligible for the award,
                // then we will not count any combinations that can be made using it.
                if (totalAbsences >= 2 || consecutiveLates >= 3)
                {
                    return 0;
                }
                // If we have generated a combination of length 'n' we will count it.
                if (n == 0)
                {
                    return 1;
                }
                // If we have already seen this sub-problem earlier, we return the stored result.
                if (memo[n][totalAbsences][consecutiveLates] != -1)
                {
                    return memo[n][totalAbsences][consecutiveLates];
                }

                int count = 0;
                // We choose 'P' for the current position.
                count = EligibleCombinations(n - 1, totalAbsences, 0) % MOD;
                // We choose 'A' for the current position.
                count = (count + EligibleCombinations(n - 1, totalAbsences + 1, 0)) % MOD;
                // We choose 'L' for the current position.
                count = (count + EligibleCombinations(n - 1, totalAbsences, consecutiveLates + 1)) % MOD;

                // Return and store the current sub-problem result in the cache.
                return memo[n][totalAbsences][consecutiveLates] = count;
            }

            /*
            Approach 2: Bottom-Up Dynamic Programming
Complexity Analysis
•	Time complexity: O(n)
o	We iterate over n×2×3 sub-problems using nested for loops.
o	Thus, this approach will take O(6⋅n)=O(n) time.
•	Space complexity: O(n)
o	We initialized an additional array of size n⋅2⋅3.
o	Thus, this approach will take O(6⋅n)=O(n) space.

            */
            public int BottomUpWithDP(int n)
            {
                int MOD = 1000000007;
                // Cache to store sub-problem results.
                int[][][] dp = new int[n + 1][][];

                // Base case: there is 1 string of length 0 with zero 'A' and zero 'L'.
                dp[0][0][0] = 1;

                // Iterate on smaller sub-problems and use the current smaller sub-problem
                // to generate results for bigger sub-problems.
                for (int len = 0; len < n; ++len)
                {
                    for (int totalAbsences = 0; totalAbsences <= 1; ++totalAbsences)
                    {
                        for (
                            int consecutiveLates = 0;
                            consecutiveLates <= 2;
                            ++consecutiveLates
                        )
                        {
                            // Store the count when 'P' is chosen.
                            dp[len + 1][totalAbsences][0] = (dp[len +
                                    1][totalAbsences][0] +
                                dp[len][totalAbsences][consecutiveLates]) %
                            MOD;
                            // Store the count when 'A' is chosen.
                            if (totalAbsences < 1)
                            {
                                dp[len + 1][totalAbsences + 1][0] = (dp[len +
                                        1][totalAbsences + 1][0] +
                                    dp[len][totalAbsences][consecutiveLates]) %
                                MOD;
                            }
                            // Store the count when 'L' is chosen.
                            if (consecutiveLates < 2)
                            {
                                dp[len + 1][totalAbsences][consecutiveLates + 1] =
                                    (dp[len + 1][totalAbsences][consecutiveLates + 1] +
                                        dp[len][totalAbsences][consecutiveLates]) %
                                    MOD;
                            }
                        }
                    }
                }

                // Sum up the counts for all combinations of length 'n' with different absent and late counts.
                int count = 0;
                for (int totalAbsences = 0; totalAbsences <= 1; ++totalAbsences)
                {
                    for (
                        int consecutiveLates = 0;
                        consecutiveLates <= 2;
                        ++consecutiveLates
                    )
                    {
                        count = (count + dp[n][totalAbsences][consecutiveLates]) % MOD;
                    }
                }
                return count;
            }
            /*
            Approach 3: Bottom-Up Dynamic Programming, Space Optimized 
            Complexity Analysis
•	Time complexity: O(n)
o	We iterate over 2×3×n states once using the nested for-loops.
o	Thus, this approach will take O(6⋅n)=O(n) time.
•	Space complexity: O(1)
o	We use two 2×3 arrays and a handful of variables. Since the space used is not affected by the size of n, we only use constant space in this approach.

            */
            public int BottomUpWithDPSpaceOptimal(int n)
            {
                int MOD = 1000000007;
                // Cache to store current sub-problem results.
                int[][] currentStateDP = new int[2][];
                currentStateDP[0] = new int[3];
                currentStateDP[1] = new int[3];

                // Cache to store next sub-problem results.
                int[][] nextStateDP = new int[2][];
                nextStateDP[0] = new int[3];
                nextStateDP[1] = new int[3];

                // Base case: there is 1 string of length 0 with zero 'A' and zero 'L'.
                currentStateDP[0][0] = 1;

                // Iterate on smaller sub-problems and use the current smaller sub-problem
                // to generate results for bigger sub-problems.
                for (int length = 0; length < n; ++length)
                {
                    for (int totalAbsences = 0; totalAbsences <= 1; ++totalAbsences)
                    {
                        for (int consecutiveLates = 0; consecutiveLates <= 2; ++consecutiveLates)
                        {
                            // Store the count when 'P' is chosen.
                            nextStateDP[totalAbsences][0] =
                                (nextStateDP[totalAbsences][0] +
                                currentStateDP[totalAbsences][consecutiveLates]) % MOD;
                            // Store the count when 'A' is chosen.
                            if (totalAbsences < 1)
                            {
                                nextStateDP[totalAbsences + 1][0] =
                                    (nextStateDP[totalAbsences + 1][0] +
                                    currentStateDP[totalAbsences][consecutiveLates]) % MOD;
                            }
                            // Store the count when 'L' is chosen.
                            if (consecutiveLates < 2)
                            {
                                nextStateDP[totalAbsences][consecutiveLates + 1] =
                                    (nextStateDP[totalAbsences][consecutiveLates + 1] +
                                    currentStateDP[totalAbsences][consecutiveLates]) % MOD;
                            }
                        }
                    }

                    // Next state sub-problems will become current state sub-problems in the next iteration.
                    Array.Copy(nextStateDP, currentStateDP, currentStateDP.Length);
                    // Next state sub-problem results will reset to zero.
                    nextStateDP = new int[2][];
                    nextStateDP[0] = new int[3];
                    nextStateDP[1] = new int[3];
                }

                // Sum up the counts for all combinations of length 'n' with different absent and late counts.
                int totalCount = 0;
                for (int totalAbsences = 0; totalAbsences <= 1; ++totalAbsences)
                {
                    for (int consecutiveLates = 0; consecutiveLates <= 2; ++consecutiveLates)
                    {
                        totalCount = (totalCount + currentStateDP[totalAbsences][consecutiveLates]) % MOD;
                    }
                }
                return totalCount;
            }


        }


        /*
        554. Brick Wall
        https://leetcode.com/problems/brick-wall/description/

        */
        public class LeastBricksSol
        {
            /*
            Approach #1 Brute Force [Time Limit Exceeded]
            Complexity Analysis
•	Time complexity : O(n∗m). We traverse over the width(n) of the wall m times, where m is the height of the wall.
•	Space complexity : O(m). pos array of size m is used, where m is the height of the wall.

            */
            public int Naive1(List<List<int>> wall)
            {
                int[] position = new int[wall.Count];
                int count = 0, totalLength = 0, result = int.MaxValue;

                foreach (int length in wall[0])
                {
                    totalLength += length;
                }

                while (totalLength != 0)
                {
                    int brickCount = 0;
                    for (int i = 0; i < wall.Count; i++)
                    {
                        List<int> row = wall[i];
                        if (position[i] < row.Count && row[position[i]] != 0)
                        {
                            brickCount++;
                        }
                        else
                        {
                            position[i]++;
                        }
                        if (position[i] < row.Count)
                        {
                            row[position[i]]--;
                        }
                    }
                    totalLength--;
                    result = Math.Min(result, brickCount);
                }
                return result;
            }
            /*
            Approach #2 Better Brute Force[Time LImit Exceeded]
Complexity Analysis
•	Time complexity : O(n∗m). In worst case, we traverse over the length(n) of the wall m times, where m is the height of the wall.
•	Space complexity : O(m). pos array of size m is used, where m is the height of the wall.

            */
            public int Naive2(List<List<int>> wall)
            {
                int[] pos = new int[wall.Count];
                int sum = 0, res = int.MaxValue;
                foreach (int el in wall[0])
                    sum += el;
                while (sum != 0)
                {
                    int count = 0, mini = int.MaxValue;
                    for (int i = 0; i < wall.Count; i++)
                    {
                        List<int> row = wall[i];
                        if (row[pos[i]] != 0)
                        {
                            count++;
                        }
                        else
                            pos[i]++;
                        mini = Math.Min(mini, row[pos[i]]);
                    }
                    for (int i = 0; i < wall.Count; i++)
                    {
                        List<int> row = wall[i];
                        row[pos[i]] = row[pos[i] - mini];
                    }
                    sum -= mini;
                    res = Math.Min(res, count);
                }
                return res;
            }
            /*
            Approach #3 Using HashMap 
Complexity Analysis**
•	Time complexity : O(n). We traverse over the complete bricks only once. n is the total number of bricks in a wall.
•	Space complexity : O(m). map will contain at most m entries, where m refers to the width of the wall.

            */
            public int UsingHashMap(List<List<int>> wall)
            {
                Dictionary<int, int> map = new Dictionary<int, int>();
                foreach (List<int> row in wall)
                {
                    int sum = 0;
                    for (int i = 0; i < row.Count - 1; i++)
                    {
                        sum += row[i];
                        if (map.ContainsKey(sum))
                            map.Add(sum, map[sum] + 1);
                        else
                            map.Add(sum, 1);
                    }
                }
                int res = wall.Count;
                foreach (int key in map.Keys)
                    res = Math.Min(res, wall.Count - map[key]);
                return res;
            }










        }

        /*
        568. Maximum Vacation Days
        https://leetcode.com/problems/maximum-vacation-days/description/	

        */
        public class MaxVacationDaysSol
        {
            /*            
Approach #1 Using Depth First Search [Time Limit Exceeded]
Complexity Analysis
•	Time complexity : O(nk). Depth of Recursion tree will be k and each node contains n branches in the worst case. Here n represents the number of cities and k is the total number of weeks.
•	Space complexity : O(k). The depth of the recursion tree is k.

            */
            public int DFS(int[][] flights, int[][] days)
            {
                return DFSRec(flights, days, 0, 0);
            }

            public int DFSRec(int[][] flights, int[][] days, int currentCity, int weekNumber)
            {
                if (weekNumber == days[0].Length)
                    return 0;

                int maxVacation = 0;
                for (int i = 0; i < flights.Length; i++)
                {
                    if (flights[currentCity][i] == 1 || i == currentCity)
                    {
                        int vacation = days[i][weekNumber] + DFSRec(flights, days, i, weekNumber + 1);
                        maxVacation = Math.Max(maxVacation, vacation);
                    }
                }
                return maxVacation;
            }

            /*
            Approach #2 Using DFS with memoization 
            Complexity Analysis
•	Time complexity : O(n^2*k). memo array of size n∗k is filled and each cell filling takes O(n) time .
•	Space complexity : O(n∗k). memo array of size n∗k is used. Here n represents the number of cities and k is the total number of weeks.

            */
            public int DFSWithMemo(int[][] flights, int[][] days)
            {
                int[][] memo = new int[flights.Length][];
                foreach (int[] l in memo)
                    Array.Fill(l, int.MinValue);
                return DFSRec(flights, days, 0, 0, memo);
            }
            public int DFSRec(int[][] flights, int[][] days, int cur_city, int weekno, int[][] memo)
            {
                if (weekno == days[0].Length)
                    return 0;
                if (memo[cur_city][weekno] != int.MinValue)
                    return memo[cur_city][weekno];
                int maxvac = 0;
                for (int i = 0; i < flights.Length; i++)
                {
                    if (flights[cur_city][i] == 1 || i == cur_city)
                    {
                        int vac = days[i][weekno] + DFSRec(flights, days, i, weekno + 1, memo);
                        maxvac = Math.Max(maxvac, vac);
                    }
                }
                memo[cur_city][weekno] = maxvac;
                return maxvac;
            }

            /*
            Approach #3 Using 2-D Dynamic Programming
           Complexity Analysis
•	Time complexity : O(n^2*k). dp array of size n∗k is filled and each cell filling takes O(n) time. Here n represents the number of cities and k is the total number of weeks.
•	Space complexity : O(n∗k). dp array of size n∗k is used.
 
            */
            public int TwoDArrayDP(int[][] flights, int[][] days)
            {
                if (days.Length == 0 || flights.Length == 0) return 0;
                int[][] dp = new int[days.Length][];
                for (int week = days[0].Length - 1; week >= 0; week--)
                {
                    for (int cur_city = 0; cur_city < days.Length; cur_city++)
                    {
                        dp[cur_city][week] = days[cur_city][week] + dp[cur_city][week + 1];
                        for (int dest_city = 0; dest_city < days.Length; dest_city++)
                        {
                            if (flights[cur_city][dest_city] == 1)
                            {
                                dp[cur_city][week] = Math.Max(days[dest_city][week] + dp[dest_city][week + 1], dp[cur_city][week]);
                            }
                        }
                    }
                }
                return dp[0][0];
            }

            /*
            Approach #4 Using 1-D Dynamic Programming
            Complexity Analysis
•	Time complexity : O(n^2*k). dp array of size n∗k is filled and each cell filling takes O(n) time. Here n represents the number of cities and k is the total number of weeks.
•	Space complexity : O(k). dp array of size nk is used.

            */
            public int OneDArrayDP(int[][] flights, int[][] days)
            {
                if (days.Length == 0 || flights.Length == 0) return 0;
                int[] dp = new int[days.Length];
                for (int week = days[0].Length - 1; week >= 0; week--)
                {
                    int[] temp = new int[days.Length];
                    for (int cur_city = 0; cur_city < days.Length; cur_city++)
                    {
                        temp[cur_city] = days[cur_city][week] + dp[cur_city];
                        for (int dest_city = 0; dest_city < days.Length; dest_city++)
                        {
                            if (flights[cur_city][dest_city] == 1)
                            {
                                temp[cur_city] = Math.Max(days[dest_city][week] + dp[dest_city], temp[cur_city]);
                            }
                        }
                    }
                    dp = temp;
                }

                return dp[0];
            }

        }

        /*
        787. Cheapest Flights Within K Stops
        https://leetcode.com/problems/cheapest-flights-within-k-stops/description/	

        */
        public class FindCheapestPriceWithKStopsSol
        {
            /*
            Approach 1: Breadth First Search
            Complexity Analysis
Let E be the number of flights and N be the number of cities.
•	Time complexity: O(N+E⋅K)
o	Depending on improvements in the shortest distance for each node, we may process each edge multiple times. However, the maximum number of times an edge can be processed is limited by K because that is the number of levels we will investigate in this algorithm. In the worst case, this takes O(E⋅K) time. We also need O(E) to initialize the adjacency list and O(N) to initialize the dist array.
•	Space complexity: O(N+E⋅K)
o	We are processing at most E⋅K edges, so the queue takes up O(E⋅K) space in the worst case. We also need O(E) space for the adjacency list and O(N) space for the dist array.


            */
            public int BFS(int numberOfCities, int[][] flights, int source, int destination, int maxStops)
            {
                Dictionary<int, List<int[]>> adjacencyList = new Dictionary<int, List<int[]>>();
                foreach (int[] flight in flights)
                {
                    if (!adjacencyList.ContainsKey(flight[0]))
                        adjacencyList[flight[0]] = new List<int[]>();

                    adjacencyList[flight[0]].Add(new int[] { flight[1], flight[2] });
                }

                int[] distances = new int[numberOfCities];
                Array.Fill(distances, int.MaxValue);

                Queue<int[]> queue = new Queue<int[]>();
                queue.Enqueue(new int[] { source, 0 });
                int stops = 0;

                while (stops <= maxStops && queue.Count > 0)
                {
                    int size = queue.Count;
                    // Iterate on current level.
                    while (size-- > 0)
                    {
                        int[] current = queue.Dequeue();
                        int currentNode = current[0];
                        int currentDistance = current[1];

                        if (!adjacencyList.ContainsKey(currentNode))
                            continue;
                        // Loop over neighbors of popped node.
                        foreach (int[] neighbor in adjacencyList[currentNode])
                        {
                            int neighborCity = neighbor[0];
                            int price = neighbor[1];
                            if (price + currentDistance >= distances[neighborCity])
                                continue;
                            distances[neighborCity] = price + currentDistance;
                            queue.Enqueue(new int[] { neighborCity, distances[neighborCity] });
                        }
                    }
                    stops++;
                }
                return distances[destination] == int.MaxValue ? -1 : distances[destination];
            }

            /*
            
Approach 2: Bellman Ford
Complexity Analysis
Let E be the number of flights and N be number of cities.
•	Time complexity: O((N+E)⋅K)
o	We are iterating over all the edges K+1 times which takes O(E⋅K). At the start and end of each iteration, we also swap distance arrays, which take O(N⋅K) time for all the iterations. This gives us a time complexity of O(E⋅K+N⋅K)=O((N+E)⋅K)
•	Space complexity: O(N)
o	We are using dist and temp arrays, which each require O(N) space.

            */
            public int BellmanFord(int n, int[][] flights, int src, int dst, int k)
            {
                // Distance from source to all other nodes.
                int[] dist = new int[n];
                Array.Fill(dist, int.MaxValue);
                dist[src] = 0;

                // Run only K+1 times since we want shortest distance in K hops
                for (int i = 0; i <= k; i++)
                {
                    // Create a copy of dist vector.
                    int[] temp = new int[n];
                    Array.Copy(dist, temp, n);
                    foreach (int[] flight in flights)
                    {
                        if (dist[flight[0]] != int.MaxValue)
                        {
                            temp[flight[1]] = Math.Min(temp[flight[1]], dist[flight[0]] + flight[2]);
                        }
                    }
                    // Copy the temp vector into dist.
                    dist = temp;
                }
                return dist[dst] == int.MaxValue ? -1 : dist[dst];
            }

            /*
            Approach 3: Dijkstra With PQ
            Complexity Analysis
          Let E be the number of flights and N be number of cities in the given problem.
          •	Time complexity: O(N+E⋅K⋅log(E⋅K))
          o	Let's assume any node A is popped out of the queue in an iteration. If the steps taken to visit A are more than stops[node], we do not iterate over the neighbors of A. However, we will iterate over neighbors of A if the steps are less than stops[A], which can be true K times. A can be popped the first time with K steps, followed by K-1 steps, and so on until 1 step. The same argument would be valid for any other node like A. As a result, each edge can only be processed K times, resulting in O(E⋅K) elements being processed.
          o	It will take the priority queue O(E⋅K⋅log(E⋅K)) time to push or pop O(E⋅K) elements.
          o	We've added O(N) time by using the stops array.
          •	Space complexity: O(N+E⋅K)
          o	We are using the adj array, which requires O(E) memory. The stop array would require O(N) memory. As previously stated, the priority queue can only have O(E⋅K) elements.

            */
            public int DijkstraWithPQ(int numberOfCities, int[][] flights, int source, int destination, int maxStops)
            {
                Dictionary<int, List<int[]>> adjacencyList = new Dictionary<int, List<int[]>>();

                foreach (int[] flight in flights)
                {
                    if (!adjacencyList.ContainsKey(flight[0]))
                        adjacencyList[flight[0]] = new List<int[]>();

                    adjacencyList[flight[0]].Add(new int[] { flight[1], flight[2] });
                }
                int[] stops = new int[numberOfCities];
                Array.Fill(stops, int.MaxValue);
                PriorityQueue<int[], int> priorityQueue = new PriorityQueue<int[], int>(Comparer<int>.Create((a, b) => a - b));
                // {distance_from_source_node, node, number_of_stops_from_source_node}
                priorityQueue.Enqueue(new int[] { 0, source, 0 }, 0);

                while (priorityQueue.Count > 0)
                {
                    int[] temp = priorityQueue.Dequeue();
                    int distance = temp[0];
                    int currentNode = temp[1];
                    int steps = temp[2];
                    // We have already encountered a path with a lower cost and fewer stops,
                    // or the number of stops exceeds the limit.
                    if (steps > stops[currentNode] || steps > maxStops + 1)
                        continue;
                    stops[currentNode] = steps;
                    if (currentNode == destination)
                        return distance;
                    if (!adjacencyList.ContainsKey(currentNode))
                        continue;
                    foreach (int[] flight in adjacencyList[currentNode])
                    {
                        priorityQueue.Enqueue(new int[] { distance + flight[1], flight[0], steps + 1 }, distance + flight[1]);
                    }
                }
                return -1;
            }

        }

        /*
        2093. Minimum Cost to Reach City With Discounts
        https://leetcode.com/problems/minimum-cost-to-reach-city-with-discounts/
        */
        public class MinimumCostToReachCityWithDiscountsSol
        {
            /*
            Approach 1: Dijkstra's Algorithm using Priority Queue
            Complexity Analysis
Let N be the number of nodes (cities) and E be the number of edges (highways). Let K be the number of discounts.
•	Time Complexity: O((N⋅K+E)⋅log(N⋅K))
Constructing the graph representation involves iterating over the highways array, where each highway is processed in constant time. Thus, this step takes O(E) time.
The priority queue operations involve inserting and extracting elements. Each element can be inserted and extracted up to N⋅(K+1) times, and each operation takes O(log(N⋅(K+1))) time. Therefore, the priority queue operations take O((N⋅K)log(N⋅K)) time.
The relaxation of edges happens for each node and each discount state, leading to O(E⋅K) relaxation operations. Each operation involves updating the priority queue, taking O(log(N⋅K)) time. Thus, relaxation operations take O(Elog(N⋅K)) time.
Combining these steps, the overall time complexity is:
O(E+(N⋅K)log(N⋅K)+Elog(N⋅K))=O((N⋅K+E)⋅log(N⋅K))
•	Space Complexity: O(N⋅K+E)
The graph is stored as an adjacency list, requiring space equal to the number of highways E.
The distance table dist and the visited arrays are 2D arrays of size N×(K+1).
The priority queue used can contain up to N×(K+1) elements in the worst case, corresponding to each city with each possible number of discounts used.
Therefore, the space complexity is O(E+N⋅K)

            */
            public int DijkstraWithPQ(int numberOfCities, int[][] highways, int availableDiscounts)
            {
                // Construct the graph from the given highways array
                List<List<int[]>> graph = new List<List<int[]>>();
                for (int i = 0; i < numberOfCities; ++i)
                {
                    graph.Add(new List<int[]>());
                }
                foreach (int[] highway in highways)
                {
                    int startCity = highway[0], endCity = highway[1], toll = highway[2];
                    graph[startCity].Add(new int[] { endCity, toll });
                    graph[endCity].Add(new int[] { startCity, toll });
                }

                // Min-heap priority queue to store tuples of (cost, city, discounts used)
                PriorityQueue<(int cost, int city, int discountsUsed), int> priorityQueue = new PriorityQueue<(int cost, int city, int discountsUsed), int>(
                    Comparer<int>.Create((a, b) => a.CompareTo(b))
                );
                priorityQueue.Enqueue((0, 0, 0), 0); // Start from city 0 with cost 0 and 0 discounts used

                // 2D array to track minimum distance to each city with a given number of discounts used
                int[][] distance = new int[numberOfCities][];
                for (int i = 0; i < numberOfCities; i++)
                {
                    distance[i] = new int[availableDiscounts + 1];
                    Array.Fill(distance[i], int.MaxValue);
                }
                distance[0][0] = 0;

                bool[][] visited = new bool[numberOfCities][];
                for (int i = 0; i < numberOfCities; i++)
                {
                    visited[i] = new bool[availableDiscounts + 1];
                }

                while (priorityQueue.Count > 0)
                {
                    var current = priorityQueue.Dequeue();
                    int currentCost = current.cost, currentCity = current.city, discountsUsed = current.discountsUsed;

                    // Skip processing if already visited with the same number of discounts used
                    if (visited[currentCity][discountsUsed])
                    {
                        continue;
                    }
                    visited[currentCity][discountsUsed] = true;

                    // Explore all neighbors of the current city
                    foreach (int[] neighbor in graph[currentCity])
                    {
                        int nextCity = neighbor[0], toll = neighbor[1];

                        // Case 1: Move to the neighbor without using a discount
                        if (currentCost + toll < distance[nextCity][discountsUsed])
                        {
                            distance[nextCity][discountsUsed] = currentCost + toll;
                            priorityQueue.Enqueue((distance[nextCity][discountsUsed], nextCity, discountsUsed), distance[nextCity][discountsUsed]);
                        }

                        // Case 2: Move to the neighbor using a discount if available
                        if (discountsUsed < availableDiscounts)
                        {
                            int newCostWithDiscount = currentCost + toll / 2;
                            if (newCostWithDiscount < distance[nextCity][discountsUsed + 1])
                            {
                                distance[nextCity][discountsUsed + 1] = newCostWithDiscount;
                                priorityQueue.Enqueue((newCostWithDiscount, nextCity, discountsUsed + 1), newCostWithDiscount);
                            }
                        }
                    }
                }

                // Find the minimum cost to reach city n-1 with any number of discounts used
                int minCost = int.MaxValue;
                for (int d = 0; d <= availableDiscounts; ++d)
                {
                    minCost = Math.Min(minCost, distance[numberOfCities - 1][d]);
                }
                return minCost == int.MaxValue ? -1 : minCost;
            }

            /*
            Approach 2: Space Optimized Dijkstra's Algorithm
            Complexity Analysis
Let N be the number of nodes (cities) and E be the number of edges (highways). Let K be the number of discounts.
•	Time Complexity: O((N⋅K+E)⋅log(N⋅K))
Constructing the graph representation involves iterating over the highways array, where each highway is processed in constant time. Thus, this step takes O(E) time.
The priority queue operations involve inserting and extracting elements. Each element can be inserted and extracted up to N⋅(K+1) times, and each operation takes O(log(N⋅(K+1))) time. Therefore, the priority queue operations take O((N⋅K)log(N⋅K)) time.
The relaxation of edges happens for each node and each discount state, leading to O(E⋅K) relaxation operations. Each operation involves updating the priority queue, taking O(log(N⋅K)) time. Thus, relaxation operations take O(Elog(N⋅K)) time.
Combining these steps, the overall time complexity is,
O(E+(N⋅K)log(N⋅K)+Elog(N⋅K))=O((N⋅K+E)log(N⋅K))
•	Space Complexity: O(N⋅K+E)
The graph is stored as an adjacency list, requiring space equal to the number of highways E.
The distance table dist is a 2D array of size N×(K+1).
The priority queue used can contain up to N×(K+1) elements in the worst case, corresponding to each city with each possible number of discounts used.
Therefore, the space complexity is, O(E+N⋅K)

            */
            public int MinimumCost(int numberOfCities, int[][] highways, int availableDiscounts)
            {
                // Construct the graph from the given highways array
                List<List<int[]>> graph = new List<List<int[]>>();
                for (int i = 0; i < numberOfCities; ++i)
                {
                    graph.Add(new List<int[]>());
                }
                foreach (int[] highway in highways)
                {
                    int startingCity = highway[0], destinationCity = highway[1], toll = highway[2];
                    graph[startingCity].Add(new int[] { destinationCity, toll });
                    graph[destinationCity].Add(new int[] { startingCity, toll });
                }

                // Min-heap priority queue to store tuples of (cost, city, discounts used)
                PriorityQueue<(int cost, int city, int discountsUsed), int> priorityQueue = new PriorityQueue<(int cost, int city, int discountsUsed), int>(
                     Comparer<int>.Create((a, b) => a.CompareTo(b))
                 );
                priorityQueue.Enqueue((0, 0, 0), 0); // Start from city 0 with cost 0 and 0 discounts used

                // 2D array to track minimum distance to each city with a given number of discounts used
                int[][] distance = new int[numberOfCities][];
                for (int i = 0; i <= availableDiscounts; i++)
                {
                    distance[i] = new int[availableDiscounts + 1];
                    Array.Fill(distance[i], int.MaxValue);
                }
                distance[0][0] = 0;

                while (priorityQueue.Count > 0)
                {
                    (int currentCost, int currentCity, int discountsUsed) = priorityQueue.Dequeue();

                    // If this cost is already higher than the known minimum, skip it
                    if (currentCost > distance[currentCity][discountsUsed])
                    {
                        continue;
                    }

                    // Explore all neighbors of the current city
                    foreach (int[] neighbor in graph[currentCity])
                    {
                        int nextCity = neighbor[0], toll = neighbor[1];

                        // Case 1: Move to the neighbor without using a discount
                        if (currentCost + toll < distance[nextCity][discountsUsed])
                        {
                            distance[nextCity][discountsUsed] = currentCost + toll;
                            priorityQueue.Enqueue((distance[nextCity][discountsUsed], nextCity, discountsUsed), distance[nextCity][discountsUsed]);
                        }

                        // Case 2: Move to the neighbor using a discount if available
                        if (discountsUsed < availableDiscounts)
                        {
                            int newCostWithDiscount = currentCost + toll / 2;
                            if (newCostWithDiscount < distance[nextCity][discountsUsed + 1])
                            {
                                distance[nextCity][discountsUsed + 1] = newCostWithDiscount;
                                priorityQueue.Enqueue((newCostWithDiscount, nextCity, discountsUsed + 1), newCostWithDiscount);
                            }
                        }
                    }
                }

                // Find the minimum cost to reach city n-1 with any number of discounts used
                int minimumCost = int.MaxValue;
                for (int d = 0; d <= availableDiscounts; ++d)
                {
                    minimumCost = Math.Min(minimumCost, distance[numberOfCities - 1][d]);
                }
                return minimumCost == int.MaxValue ? -1 : minimumCost;
            }
        }


        /*
        1135. Connecting Cities With Minimum Cost
        https://leetcode.com/problems/connecting-cities-with-minimum-cost/description/

        */
        public class MinimumCostToConnectAllCitiesSol
        {

            /*
            Approach 1: Minimum Spanning Tree (Using Kruskal's algorithm)
            Complexity Analysis
•	Time complexity: Assuming N to be the total number of nodes (cities) and M to be the total number of edges (connections). Sorting all the M connections will take O(M⋅logM). Performing union find each time will take log∗N (Iterated logarithm). Hence for M edges, it's O(M⋅log∗N) which is practically O(M) as the value of iterated logarithm, log∗N never exceeds 5.
•	Space complexity: O(N), space required by parents and weights.

            */
            public int MinSpanTreeWithDisjointSet(int numberOfNodes, int[][] connections)
            {
                DisjointSet disjointSet = new DisjointSet(numberOfNodes);
                // Sort connections based on their weights (in increasing order)
                Array.Sort(connections, (a, b) => a[2] - b[2]);
                // Keep track of total edges added in the MST
                int totalEdges = 0;
                // Keep track of the total cost of adding all those edges
                int totalCost = 0;
                for (int i = 0; i < connections.Length; ++i)
                {
                    int nodeA = connections[i][0];
                    int nodeB = connections[i][1];
                    // Do not add the edge from nodeA to nodeB if it is already connected
                    if (disjointSet.IsInSameGroup(nodeA, nodeB)) continue;
                    // If nodeA and nodeB are not connected, take union
                    disjointSet.Union(nodeA, nodeB);
                    // increment cost
                    totalCost += connections[i][2];
                    // increment number of edges added in the MST
                    totalEdges++;
                }
                // If all numberOfNodes are connected, the MST will have a total of numberOfNodes - 1 edges
                if (totalEdges == numberOfNodes - 1)
                {
                    return totalCost;
                }
                else
                {
                    return -1;
                }
            }

            class DisjointSet
            {
                private int[] weights; // Used to store weights of each nodes 
                private int[] parents;

                public void Union(int nodeA, int nodeB)
                {
                    int rootA = Find(nodeA);
                    int rootB = Find(nodeB);
                    // If both nodeA and nodeB have same root, i.e. they both belong to the same set, hence we don't need to take union
                    if (rootA == rootB) return;

                    // Weighted union
                    if (this.weights[rootA] > this.weights[rootB])
                    {
                        // In case rootA is having more weight
                        // 1. Make rootA as the parent of rootB
                        // 2. Increment the weight of rootA by rootB's weight
                        this.parents[rootB] = rootA;
                        this.weights[rootA] += this.weights[rootB];
                    }
                    else
                    {
                        // Otherwise
                        // 1. Make rootB as the parent of rootA
                        // 2. Increment the weight of rootB by rootA's weight
                        this.parents[rootA] = rootB;
                        this.weights[rootB] += this.weights[rootA];
                    }
                }

                public int Find(int node)
                {
                    // Traverse all the way to the top (root) going through the parent nodes
                    while (node != this.parents[node])
                    {
                        // Path compression
                        // node's grandparent is now node's parent
                        this.parents[node] = this.parents[this.parents[node]];
                        node = this.parents[node];
                    }
                    return node;
                }

                public bool IsInSameGroup(int nodeA, int nodeB)
                {
                    // Return true if both nodeA and nodeB belong to the same set, otherwise return false
                    return Find(nodeA) == Find(nodeB);
                }

                // Initialize weight for each node to be 1
                public DisjointSet(int numberOfNodes)
                {
                    this.parents = new int[numberOfNodes + 1];
                    this.weights = new int[numberOfNodes + 1];
                    // Set the initial parent node to itself
                    for (int i = 1; i <= numberOfNodes; ++i)
                    {
                        this.parents[i] = i;
                        this.weights[i] = 1;
                    }
                }
            }
        }

        /*
        2247. Maximum Cost of Trip With K Highways
        https://leetcode.com/problems/maximum-cost-of-trip-with-k-highways/description/
        */
        public class MaxCostOfTripWithKHighwaysSol
        {
            /*
            Time and Space Complexity
The given Python code is an implementation of the bitmask dynamic programming algorithm for the problem of finding the maximum total cost of servicing k highways among n different cities. Here's an analysis of the time complexity and space complexity of the code:
Time Complexity
The time complexity is determined by the nested loops in the code and the operations within those loops.
•	The outer loop runs for every subset of the n cities, hence it goes through 2^n iterations as it considers every possible combination of cities.
•	Inside the outer loop, there is another loop that iterates over n cities.
•	Inside the second loop, there is an inner loop traversing the adjacency list of each city. In the worst case, this could be n-1 edges for a complete graph.
Putting it together, the worst-case time complexity is O(n * 2^n * n) = O(n^2 * 2^n).
Space Complexity
The space complexity is determined by the space used to store all the information needed during computation.
•	The 2D array f has a size of (2^n) * n, used to store the maximum costs of traveling through subsets of cities, resulting in a space complexity of O(n * 2^n).
•	The graph g stores the edges between cities. In the worst case, it can store up to n * (n - 1) / 2 edges for a complete graph. However, the space used for g is not dominant compared to f.
Therefore, the overall space complexity is O(n * 2^n) as it is the largest memory allocation in the code.

            */
            // Maximum cost calculation method, for a given number of cities (numCities), a list of highways with costs,
            // and the number of connecting highways to consider (maxHighways).
            public int DPWithBitMasking(int numCities, int[][] highways, int maxHighways)
            {
                // If maxHighways >= numCities, a valid path that uses maxHighways does not exist
                if (maxHighways >= numCities)
                {
                    return -1;
                }

                // Graph representation: an array of lists where each list contains pairs [city, cost]
                List<int[]>[] graph = new List<int[]>[numCities];
                // Initialize the adjacency lists for each city
                for (int i = 0; i < numCities; i++)
                {
                    graph[i] = new List<int[]>();
                }

                // Populate the graph with the given highways data
                foreach (int[] highway in highways)
                {
                    int cityA = highway[0], cityB = highway[1], cost = highway[2];
                    graph[cityA].Add(new int[] { cityB, cost });
                    graph[cityB].Add(new int[] { cityA, cost });
                }

                // Dynamic programming matrix to keep track of max costs f[state][city]
                int[][] dp = new int[1 << numCities][];

                // Initializing the dp matrix with minimum values
                for (int state = 0; state < dp.Length; state++)
                {
                    for (int j = 0; j < numCities; j++)
                    {
                        dp[state][j] = int.MinValue / 2; // Using MIN_VALUE/2 to prevent overflow when adding
                    }
                }

                // Base cases where each city is the only one in the set, cost is 0
                for (int i = 0; i < numCities; ++i)
                {
                    dp[1 << i][i] = 0;
                }

                // Variable to keep track of the maximum cost found
                int maxCost = -1;

                // Iterate over all sets of cities (states)
                for (int state = 0; state < (1 << numCities); ++state)
                {
                    // Iterate over all cities
                    for (int currentCity = 0; currentCity < numCities; ++currentCity)
                    {
                        // Check if the current city is included in the current state
                        if ((state & (1 << currentCity)) != 0)
                        {
                            // Explore all the highways connected to the current city 
                            foreach (int[] edge in graph[currentCity])
                            {
                                int nextCity = edge[0], nextCost = edge[1];
                                // Check if the next city is in the current state
                                if ((state & (1 << nextCity)) != 0)
                                {
                                    // Update the maximum cost for the current state and city
                                    dp[state][currentCity] = Math.Max(
                                        dp[state][currentCity],
                                        dp[state ^ (1 << currentCity)][nextCity] + nextCost
                                    );
                                }
                            }
                        }
                        // Once we have a state with exactly k + 1 cities, update the maximum cost
                        if (CountBits(state) == maxHighways + 1)
                        {
                            maxCost = Math.Max(maxCost, dp[state][currentCity]);
                        }
                    }
                }
                // Return the maximum cost found, or -1 if no path with exact k highways is found
                return maxCost;
            }

            private int CountBits(int number)
            {
                int count = 0;
                while (number > 0)
                {
                    count += number & 1;
                    number >>= 1;
                }
                return count;
            }
        }

        /*
        1928. Minimum Cost to Reach Destination in Time
        https://leetcode.com/problems/minimum-cost-to-reach-destination-in-time/description/

        */
        public class MinCostToReachDestInTimeSol
        {
            /*
            Approach: Dijkstra with Priority Queue
            TC : O(V*Vlog V)
            */
            public int DijkstraWithPQ(int maxTime, int[][] edges, int[] passingFees)
            {
                Dictionary<int, List<int[]>> adjacencyMap = new Dictionary<int, List<int[]>>();
                foreach (int[] edge in edges)
                {
                    int fromNode = edge[0];
                    int toNode = edge[1];
                    int travelTime = edge[2];

                    if (!adjacencyMap.ContainsKey(fromNode))
                    {
                        adjacencyMap[fromNode] = new List<int[]>();
                    }
                    if (!adjacencyMap.ContainsKey(toNode))
                    {
                        adjacencyMap[toNode] = new List<int[]>();
                    }

                    adjacencyMap[fromNode].Add(new int[] { toNode, travelTime });
                    adjacencyMap[toNode].Add(new int[] { fromNode, travelTime });
                }

                PriorityQueue<int[], int[]> priorityQueue = new PriorityQueue<int[], int[]>(
                    Comparer<int[]>.Create((a, b) => a[1] == b[1] ? a[2] - b[2] : a[1] - b[1]));
                var nodeCodeTime = new int[] { 0, passingFees[0], 0 };
                priorityQueue.Enqueue(nodeCodeTime, nodeCodeTime); // node cost time

                int numberOfNodes = passingFees.Length;
                int[] distance = new int[numberOfNodes];
                int[] times = new int[numberOfNodes];

                Array.Fill(distance, int.MaxValue);
                Array.Fill(times, int.MaxValue);
                distance[0] = 0;
                times[0] = 0;

                while (priorityQueue.Count > 0)
                {
                    int[] current = priorityQueue.Dequeue();
                    int currentNode = current[0];
                    int currentCost = current[1];
                    int currentTime = current[2];

                    if (currentTime > maxTime)
                    {
                        continue;
                    }

                    if (currentNode == numberOfNodes - 1) return currentCost;

                    foreach (int[] neighbor in adjacencyMap.GetValueOrDefault(currentNode, new List<int[]>()))
                    {
                        int neighborNode = neighbor[0];
                        int neighborCost = passingFees[neighborNode];

                        if (currentCost + neighborCost < distance[neighborNode])
                        {
                            distance[neighborNode] = currentCost + neighborCost;
                            var tmp = new int[] { neighborNode, currentCost + neighborCost, currentTime + neighbor[1] };
                            priorityQueue.Enqueue(tmp, tmp);
                            times[neighborNode] = currentTime + neighbor[1];
                        }

                        else if (currentTime + neighbor[1] < times[neighborNode])
                        {
                            var tmp = new int[] { neighborNode, currentCost + neighborCost, currentTime + neighbor[1] };
                            priorityQueue.Enqueue(tmp, tmp);
                            times[neighborNode] = currentTime + neighbor[1];
                        }

                    }
                }

                return distance[numberOfNodes - 1] == int.MaxValue || times[numberOfNodes - 1] > maxTime ? -1 : distance[numberOfNodes - 1];
            }
        }

        /*
        573. Squirrel Simulation
        https://leetcode.com/problems/squirrel-simulation/description/
        */
        public class SquirrelSimulationSol
        {
            /*
Approach 1: Simple Solution
Complexity Analysis
•	Time complexity : O(n). We need to traverse over the whole nuts array once. n refers to the size of nuts array.
•	Space complexity : O(1). Constant space is used.
 
            */
            public int SimpleSol(int height, int width, int[] tree, int[] squirrel, int[][] nuts)
            {
                int tot_dist = 0, d = int.MinValue;
                foreach (int[] nut in nuts)
                {
                    tot_dist += (Distance(nut, tree) * 2);
                    d = Math.Max(d, Distance(nut, tree) - Distance(nut, squirrel));
                }
                return tot_dist - d;
            }
            public int Distance(int[] a, int[] b)
            {
                return Math.Abs(a[0] - b[0]) + Math.Abs(a[1] - b[1]);
            }
        }

        /*
        575. Distribute Candies
        https://leetcode.com/problems/distribute-candies/description/	
        */
        public class DistributeCandiesSol
        {
            /*
            Approach 1: Brute Force
            Complexity Analysis
Let N be the the length of candyType.
•	Time complexity : O(N^2). We traverse over each of the N elements of candyType, and for each, we check all of the elements before it. Checking each item for each item is the classic O(N^2) time complexity pattern.
•	Space complexity : O(1). We don't allocate any additional data structures, instead only using constant space variables.

            */
            public int Naive(int[] candyType)
            {
                // Initiate a variable to count how many unique candies are in the array.
                int uniqueCandies = 0;
                // For each candy, we're going to check whether or not we've already
                // seen a candy identical to it.
                for (int i = 0; i < candyType.Length; i++)
                {
                    // Start by assuming that the candy IS unique.
                    bool isUnique = true;
                    // Check each candy BEFORE this candy.
                    for (int j = 0; j < i; j++)
                    {
                        // If this candy is the same as a previous one, it isn't unique.
                        if (candyType[i] == candyType[j])
                        {
                            isUnique = false;
                            break;
                        }
                    }
                    if (isUnique)
                    {
                        uniqueCandies++;
                    }
                }
                // The answer is the minimum out of the number of unique candies, and 
                // half the length of the candyType array.
                return Math.Min(uniqueCandies, candyType.Length / 2);
            }
            /*
            Approach 2: Sorting
Complexity Analysis
Let N be the the length of candyType.
•	Time complexity : O(NlogN).
We start by sorting the N elements in candyType, which has a cost of O(NlogN).
We then perform a single pass through candyType, performing an O(1) operation at each step: this has a total cost of O(N).
This gives us a total of O(NlogN)+O(N). When adding complexities, we only keep the one that is strictly bigger, this leaves us with O(NlogN).
•	Space complexity : Dependent on the sorting algorithm implementation, which is generally between O(1) and O(N).
Python and Java now use Timsort, which requires O(N) space.
The heapify variant for Python is O(1), as it uses Heapsort.

            */
            public int Sorting(int[] candyType)
            {
                // We start by sorting candyType.
                Array.Sort(candyType);
                // The first candy is always unique.
                int uniqueCandies = 1;
                // For each candy, starting from the second candy...
                for (int i = 1; i < candyType.Length && uniqueCandies < candyType.Length / 2; i++)
                {
                    // This candy is unique if it is different to the one
                    // immediately before it.
                    if (candyType[i] != candyType[i - 1])
                    {
                        uniqueCandies++;
                    }
                }
                // Like before, the answer is the minimum out of the number of unique 
                // candies, and half the length of the candyType array.
                return Math.Min(uniqueCandies, candyType.Length / 2);
            }

            /*
            Approach 3: Using a Hash Set
           Complexity Analysis
    Let N be the the length of candyType.
    •	Time complexity : O(N).
    Adding an item into a Hash Set has an amortized time of O(1). Therefore, adding N items requires O(N) time. All of the other operations we use are O(1).
    •	Space complexity : O(N).
    The worst case for space complexity occurs when all N elements are unique. This will result in a Hash Set containing N elements.

            */
            public int HashSet(int[] candyType)
            {
                // Create an empty Hash Set, and add each candy into it.
                HashSet<int> uniqueCandiesSet = new HashSet<int>();
                foreach (int candy in candyType)
                {
                    uniqueCandiesSet.Add(candy);
                }
                // Then, find the answer in the same way as before.
                return Math.Min(uniqueCandiesSet.Count, candyType.Length / 2);
            }
        }
        /*
        582. Kill Process
       https://leetcode.com/problems/kill-process/description/
        */
        public class KillProcessSol
        {
            /*
            Approach #1 Depth First Search [Time Limit Exceeded]
Complexity Analysis
•	Time complexity : O(n^n). O(n^n) function calls will be made in the worst case
•	Space complexity : O(n). The depth of the recursion tree can go upto n.

            */
            public IList<int> DFS(List<int> processIds, List<int> parentProcessIds, int killProcessId)
            {
                List<int> processList = new List<int>();
                if (killProcessId == 0)
                    return processList;
                processList.Add(killProcessId);
                for (int i = 0; i < parentProcessIds.Count; i++)
                {
                    if (parentProcessIds[i] == killProcessId)
                    {
                        processList.AddRange(DFS(processIds, parentProcessIds, processIds[i]));
                    }
                }
                return processList;
            }
            /*        
    Approach #2 Tree Simulation [Accepted]
    Complexity Analysis
    •	Time complexity : O(n). We need to traverse over the ppid and pid array of size n once. The getAllChildren function also takes at most n time, since no node can be a child of two nodes.
    •	Space complexity : O(n). map of size n is used.
            */
            public class Node
            {
                public int Value;
                public List<Node> Children = new List<Node>();
            }

            public List<int> TreeSimulation(List<int> processIds, List<int> parentProcessIds, int killProcessId)
            {
                Dictionary<int, Node> processMap = new Dictionary<int, Node>();

                foreach (int id in processIds)
                {
                    Node node = new Node();
                    node.Value = id;
                    processMap[id] = node;
                }

                for (int i = 0; i < parentProcessIds.Count; i++)
                {
                    if (parentProcessIds[i] > 0)
                    {
                        Node parentNode = processMap[parentProcessIds[i]];
                        parentNode.Children.Add(processMap[processIds[i]]);
                    }
                }

                List<int> resultList = new List<int>();
                resultList.Add(killProcessId);
                GetAllChildren(processMap[killProcessId], resultList);
                return resultList;
            }

            private void GetAllChildren(Node parentNode, List<int> resultList)
            {
                foreach (Node childNode in parentNode.Children)
                {
                    resultList.Add(childNode.Value);
                    GetAllChildren(childNode, resultList);
                }
            }

            /*
            Approach #3 HashMap + Depth First Search [Accepted]
            Complexity Analysis
    •	Time complexity : O(n). We need to traverse over the ppid array of size n once. The getAllChildren function also takes at most n time, since no node can be a child of two nodes.
    •	Space complexity : O(n). map of size n is used.

            */
            public List<int> DFSWithHashMap(List<int> processIds, List<int> parentProcessIds, int processToKill)
            {
                Dictionary<int, List<int>> processMap = new Dictionary<int, List<int>>();
                for (int i = 0; i < parentProcessIds.Count; i++)
                {
                    if (parentProcessIds[i] > 0)
                    {
                        List<int> children = processMap.GetValueOrDefault(parentProcessIds[i], new List<int>());
                        children.Add(processIds[i]);
                        processMap[parentProcessIds[i]] = children;
                    }
                }
                List<int> result = new List<int>();
                result.Add(processToKill);
                GetAllChildren(processMap, result, processToKill);
                return result;
            }

            public void GetAllChildren(Dictionary<int, List<int>> processMap, List<int> result, int processToKill)
            {
                if (processMap.ContainsKey(processToKill))
                {
                    foreach (int childId in processMap[processToKill])
                    {
                        result.Add(childId);
                        GetAllChildren(processMap, result, childId);
                    }
                }
            }
            /*
            Approach #4 HashMap + Breadth First Search [Accepted]:
            Complexity Analysis
•	Time complexity : O(n). We need to traverse over the ppid array of size n once. Also, at most n additions/removals are done from the queue.
•	Space complexity : O(n). map of size n is used.

            */
            public List<int> BFSWithHashMap(List<int> processIds, List<int> parentProcessIds, int killProcessId)
            {
                Dictionary<int, List<int>> processMap = new Dictionary<int, List<int>>();
                for (int i = 0; i < parentProcessIds.Count; i++)
                {
                    if (parentProcessIds[i] > 0)
                    {
                        if (!processMap.ContainsKey(parentProcessIds[i]))
                        {
                            processMap[parentProcessIds[i]] = new List<int>();
                        }
                        processMap[parentProcessIds[i]].Add(processIds[i]);
                    }
                }

                Queue<int> processQueue = new Queue<int>();
                List<int> resultList = new List<int>();
                processQueue.Enqueue(killProcessId);
                while (processQueue.Count > 0)
                {
                    int currentProcessId = processQueue.Dequeue();
                    resultList.Add(currentProcessId);
                    if (processMap.ContainsKey(currentProcessId))
                    {
                        foreach (int childProcessId in processMap[currentProcessId])
                        {
                            processQueue.Enqueue(childProcessId);
                        }
                    }
                }
                return resultList;
            }

        }

        /*
        587. Erect the Fence
        https://leetcode.com/problems/erect-the-fence/description/

        */
        public class ErectFenceSol
        {
            /*
            Approach 1: Jarvis Algorithm
            Complexity Analysis
•	Time complexity : O(m∗n). For every point on the hull we examine all the other points to determine the next point. Here n is number of input points and m is number of output or hull points (m≤n).
•	Space complexity : O(m). List hull grows upto size m.

            */
            public int[][] JarvisAlgo(int[][] points)
            {
                HashSet<int[]> hull = new HashSet<int[]>();
                if (points.Length < 4)
                {
                    foreach (int[] point in points)
                        hull.Add(point);
                    return hull.ToArray();
                }
                int leftMostIndex = 0;
                for (int i = 0; i < points.Length; i++)
                    if (points[i][0] < points[leftMostIndex][0])
                        leftMostIndex = i;
                int currentPointIndex = leftMostIndex;
                do
                {
                    int nextPointIndex = (currentPointIndex + 1) % points.Length;
                    for (int i = 0; i < points.Length; i++)
                    {
                        if (hull.Contains(points[i]))
                            continue;

                        if (CalculateOrientation(points[currentPointIndex], points[i], points[nextPointIndex]) < 0)
                        {
                            nextPointIndex = i;
                        }
                    }
                    for (int i = 0; i < points.Length; i++)
                    {
                        if (i != currentPointIndex && i != nextPointIndex && CalculateOrientation(points[currentPointIndex], points[i], points[nextPointIndex]) == 0 && IsPointBetween(points[currentPointIndex], points[i], points[nextPointIndex]))
                        {
                            hull.Add(points[i]);
                        }
                    }
                    hull.Add(points[nextPointIndex]);
                    currentPointIndex = nextPointIndex;
                }
                while (currentPointIndex != leftMostIndex);
                return hull.ToArray();
            }
            private int CalculateOrientation(int[] pointP, int[] pointQ, int[] pointR)
            {
                return (pointQ[1] - pointP[1]) * (pointR[0] - pointQ[0]) - (pointQ[0] - pointP[0]) * (pointR[1] - pointQ[1]);
            }

            private bool IsPointBetween(int[] pointP, int[] pointI, int[] pointQ)
            {
                bool isXInRange = (pointI[0] >= pointP[0] && pointI[0] <= pointQ[0]) || (pointI[0] <= pointP[0] && pointI[0] >= pointQ[0]);
                bool isYInRange = (pointI[1] >= pointP[1] && pointI[1] <= pointQ[1]) || (pointI[1] <= pointP[1] && pointI[1] >= pointQ[1]);
                return isXInRange && isYInRange;
            }

            /*        
    Approach 2: Graham Scan
    Complexity Analysis
    •	Time complexity : O(nlogn). Sorting the given points takes O(nlogn) time. Further, after sorting the points can be considered in two cases, while being pushed onto the stack or while popping from the stack. At most, every point is touched twice(both push and pop) taking 2n(O(n)) time in the worst case.
    •	Space complexity : O(n). Stack size grows upto n in worst case.

            */

            public int[][] GrahamScan(int[][] points)
            {
                if (points.Length <= 1)
                {
                    return points;
                }
                int[] bottomMost = BottomLeft(points);
                Array.Sort(points, (pointP, pointQ) =>
                {
                    double difference = Orientation(bottomMost, pointP, pointQ) - Orientation(bottomMost, pointQ, pointP);
                    if (difference == 0)
                    {
                        return Distance(bottomMost, pointP) - Distance(bottomMost, pointQ);
                    }
                    else
                    {
                        return difference > 0 ? 1 : -1;
                    }
                });

                int index = points.Length - 1;
                while (index >= 0 && Orientation(bottomMost, points[points.Length - 1], points[index]) == 0)
                {
                    index--;
                }

                for (int lower = index + 1, upper = points.Length - 1; lower < upper; lower++, upper--)
                {
                    int[] tempPoint = points[lower];
                    points[lower] = points[upper];
                    points[upper] = tempPoint;
                }

                Stack<int[]> stack = new Stack<int[]>();
                stack.Push(points[0]);
                stack.Push(points[1]);
                for (int j = 2; j < points.Length; j++)
                {
                    int[] topPoint = stack.Pop();
                    while (Orientation(stack.Peek(), topPoint, points[j]) > 0)
                    {
                        topPoint = stack.Pop();
                    }
                    stack.Push(topPoint);
                    stack.Push(points[j]);
                }
                return stack.ToArray();
            }
            public int Orientation(int[] pointP, int[] pointQ, int[] pointR)
            {
                return (pointQ[1] - pointP[1]) * (pointR[0] - pointQ[0]) - (pointQ[0] - pointP[0]) * (pointR[1] - pointQ[1]);
            }

            public int Distance(int[] pointP, int[] pointQ)
            {
                return (pointP[0] - pointQ[0]) * (pointP[0] - pointQ[0]) + (pointP[1] - pointQ[1]) * (pointP[1] - pointQ[1]);
            }

            private static int[] BottomLeft(int[][] points)
            {
                int[] bottomLeftPoint = points[0];
                foreach (int[] point in points)
                {
                    if (point[1] < bottomLeftPoint[1])
                    {
                        bottomLeftPoint = point;
                    }
                }
                return bottomLeftPoint;
            }

            /*
            Approach 3: Monotone Chain
            Complexity Analysis
            •	Time complexity : O(nlogn). Sorting the given points takes O(nlogn) time. Further, after sorting the points can be considered in two cases, while being pushed onto the hull or while popping from the hull. At most, every point is touched twice(both push and pop) taking 2n(O(n)) time in the worst case.
            •	Space complexity : O(n). hull stack can grow upto size n.

            */
            public int[][] MonotoneChain(int[][] points)
            {
                Array.Sort(points, (pointP, pointQ) =>
                {
                    return pointQ[0] - pointP[0] == 0 ? pointQ[1] - pointP[1] : pointQ[0] - pointP[0];
                });

                Stack<int[]> hull = new Stack<int[]>();
                for (int i = 0; i < points.Length; i++)
                {
                    while (hull.Count >= 2 && Orientation(hull.ToArray()[hull.Count - 2], hull.ToArray()[hull.Count - 1], points[i]) > 0)
                    {
                        hull.Pop();
                    }
                    hull.Push(points[i]);
                }
                hull.Pop();
                for (int i = points.Length - 1; i >= 0; i--)
                {
                    while (hull.Count >= 2 && Orientation(hull.ToArray()[hull.Count - 2], hull.ToArray()[hull.Count - 1], points[i]) > 0)
                    {
                        hull.Pop();
                    }
                    hull.Push(points[i]);
                }
                // remove redundant elements from the stack
                HashSet<int[]> result = new HashSet<int[]>(hull);
                return result.ToArray();
            }

        }

        /*
        1924. Erect the Fence II
        https://leetcode.com/problems/erect-the-fence-ii/description/
        */
        class ErectFenceIISol
        {
            public double[] OuterTrees(int[][] trees)
            {
                return Welzl(trees, new List<int[]>(), 0);
            }

            private double[] Welzl(int[][] points, List<int[]> boundaryPoints, int offset)
            {
                if (offset == points.Length || boundaryPoints.Count == 3)
                {
                    return Trivial(boundaryPoints);
                }

                double[] disk = Welzl(points, boundaryPoints, offset + 1);

                if (Inside(disk, points[offset]))
                {
                    return disk;
                }

                boundaryPoints.Add(points[offset]);
                disk = Welzl(points, boundaryPoints, offset + 1);
                boundaryPoints.RemoveAt(boundaryPoints.Count - 1);
                return disk;
            }

            private double[] Trivial(List<int[]> boundaryPoints)
            {
                if (boundaryPoints.Count == 0) return null;

                if (boundaryPoints.Count == 1)
                {
                    return new double[] { boundaryPoints[0][0], boundaryPoints[0][1], 0 };
                }

                if (boundaryPoints.Count == 2)
                {
                    return GetDiskFromTwoPoints(boundaryPoints[0], boundaryPoints[1]);
                }

                double[] disk01 = GetDiskFromTwoPoints(boundaryPoints[0], boundaryPoints[1]);
                if (Inside(disk01, boundaryPoints[2])) return disk01;
                double[] disk02 = GetDiskFromTwoPoints(boundaryPoints[0], boundaryPoints[2]);
                if (Inside(disk02, boundaryPoints[1])) return disk02;
                double[] disk12 = GetDiskFromTwoPoints(boundaryPoints[1], boundaryPoints[2]);
                if (Inside(disk12, boundaryPoints[0])) return disk12;

                return GetDiskFromThreePointsOnTheBoundary(boundaryPoints[0], boundaryPoints[1], boundaryPoints[2]);
            }

            private double[] GetDiskFromTwoPoints(int[] point1, int[] point2)
            {
                double x1 = point1[0], y1 = point1[1];
                double x2 = point2[0], y2 = point2[1];
                double radiusSquared = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
                return new double[] { (x1 + x2) / 2.0, (y1 + y2) / 2.0, Math.Sqrt(radiusSquared) / 2.0 };
            }

            private double[] GetDiskFromThreePointsOnTheBoundary(int[] point1, int[] point2, int[] point3)
            {
                // Find a point on the perpendicular bisector of point2 and point3 such that the distances to point1, point2, and point3 are equal.
                double[] center = GetCenter(point2[0] - point1[0], point2[1] - point1[1], point3[0] - point1[0], point3[1] - point1[1]);
                center[0] += point1[0];
                center[1] += point1[1];
                double radiusSquared = (center[0] - point1[0]) * (center[0] - point1[0]) + (center[1] - point1[1]) * (center[1] - point1[1]);
                return new double[] { center[0], center[1], Math.Sqrt(radiusSquared) };
            }

            private double[] GetCenter(double bx, double by, double cx, double cy)
            {
                double b = bx * bx + by * by;
                double c = cx * cx + cy * cy;
                double d = bx * cy - by * cx;
                return new double[] { (cy * b - by * c) / (2 * d), (bx * c - cx * b) / (2 * d) };
            }

            private bool Inside(double[] circle, int[] point)
            {
                if (circle == null) return false;
                double radiusSquared = circle[2] * circle[2];
                double distanceSquared = (circle[0] - point[0]) * (circle[0] - point[0])
                    + (circle[1] - point[1]) * (circle[1] - point[1]);
                return distanceSquared <= radiusSquared;
            }
        }

        /*
        2545. Sort the Students by Their Kth Score
        https://leetcode.com/problems/sort-the-students-by-their-kth-score/description/

        Complexity
        Time O(quick sort)
        Space O(quick sort)

        */
        public int[][] SortTheStudents(int[][] A, int k)
        {
            Array.Sort(A, (a, b) => b[k] - a[k]);
            return A;
        }

        /*
        591. Tag Validator
https://leetcode.com/problems/tag-validator/description/
        */

        public class IsValidTagNameSol
        {
            /*
            Approach 1: Stack
         Complexity Analysis
•	Time complexity : O(n). We traverse over the given code string of length n.
•	Space complexity : O(n). The stack can grow upto a size of n/3 in the worst case. e.g. In case of <A><B><C><D>, n=12 and number of tags = 12/3 = 4.
   
            */
            private Stack<string> tagStack = new Stack<string>();
            private bool hasTag = false;

            public bool UsingStack(string code)
            {
                if (code[0] != '<' || code[code.Length - 1] != '>')
                    return false;

                for (int i = 0; i < code.Length; i++)
                {
                    bool isEndingTag = false;
                    int closeIndex;

                    if (tagStack.Count == 0 && hasTag)
                        return false;

                    if (code[i] == '<')
                    {
                        if (tagStack.Count > 0 && code[i + 1] == '!')
                        {
                            closeIndex = code.IndexOf("]]>", i + 1);
                            if (closeIndex < 0 || !IsValidCdata(code.Substring(i + 2, closeIndex - (i + 2))))
                                return false;
                        }
                        else
                        {
                            if (code[i + 1] == '/')
                            {
                                i++;
                                isEndingTag = true;
                            }
                            closeIndex = code.IndexOf('>', i + 1);
                            if (closeIndex < 0 || !IsValidTagName(code.Substring(i + 1, closeIndex - (i + 1)), isEndingTag))
                                return false;
                        }
                        i = closeIndex;
                    }
                }
                return tagStack.Count == 0 && hasTag;
            }
            private bool IsValidTagName(string tagName, bool isEndingTag)
            {
                if (tagName.Length < 1 || tagName.Length > 9)
                    return false;

                for (int i = 0; i < tagName.Length; i++)
                {
                    if (!char.IsUpper(tagName[i]))
                        return false;
                }

                if (isEndingTag)
                {
                    if (tagStack.Count > 0 && tagStack.Peek().Equals(tagName))
                        tagStack.Pop();
                    else
                        return false;
                }
                else
                {
                    hasTag = true;
                    tagStack.Push(tagName);
                }
                return true;
            }

            private bool IsValidCdata(string cdata)
            {
                return cdata.IndexOf("[CDATA[") == 0;
            }

            /*
            Approach 2: Regex
          Complexity Analysis
•	Time complexity: Regular Expressions are/can be implemented in the form of finite-state machines. Thus, the time complexity is dependent on the internal representation. In the case of any suggestions, please comment below.
•	Space complexity: O(n). The stack can grow up to a size of n/3 in the worst case. e.g. In case of <A><B><C><D>, n=12 and number of tags = 12/3 = 4.
  
            */
            public bool UsingRegEx(string code)
            {
                string regexPattern = "<[A-Z]{0,9}>([^<]*(<((\\/?[A-Z]{1,9}>)|(!\\[CDATA\\[(.*?)]]>)))?)*";
                if (!Regex.IsMatch(code, regexPattern))
                    return false;

                for (int i = 0; i < code.Length; i++)
                {
                    bool isEndingTag = false;
                    if (tagStack.Count == 0 && hasTag)
                        return false;

                    if (code[i] == '<')
                    {
                        if (code[i + 1] == '!')
                        {
                            i = code.IndexOf("]]>", i + 1);
                            continue;
                        }
                        if (code[i + 1] == '/')
                        {
                            i++;
                            isEndingTag = true;
                        }
                        int closeIndex = code.IndexOf('>', i + 1);
                        if (closeIndex < 0 || !IsValidTagName(code.Substring(i + 1, closeIndex - i - 1), isEndingTag))
                            return false;

                        i = closeIndex;
                    }
                }
                return tagStack.Count == 0;

                bool IsValidTagName(string tagName, bool isEndingTag)
                {
                    if (isEndingTag)
                    {
                        if (tagStack.Count > 0 && tagStack.Peek() == tagName)
                            tagStack.Pop();
                        else
                            return false;
                    }
                    else
                    {
                        hasTag = true;
                        tagStack.Push(tagName);
                    }
                    return true;
                }
            }
        }

        /*
        616. Add Bold Tag in String
        https://leetcode.com/problems/add-bold-tag-in-string/description/

        */
        public class AddBoldTagSol
        {
            /*
            Approach: Mark Bold Characters
Complexity Analysis
Let n be s.length, m be words.length, and k be the average length of the words.
The time complexity may differ between languages. It is dependent on how the built-in method is implemented.
For example, Java's indexOf() costs O(n⋅k). The C++ standard doesn't specify implementation details, but some implementations of find() may use the KMP algorithm which can achieve O(n+k) or even O(n) in certain cases.
For this analysis, we will assume that we are using Java.
•	Time complexity: O(m⋅(n2⋅k−n⋅k2))
To calculate bold, we iterate over words. For each word, we use the built-in string finding method, which costs O(n⋅k). However, we may call it multiple times per word. In the worst case scenario, such as s = "aaaaa...aaaaa" and word = "aaaaaa", it may be called O(n−k) times. Note that this scenario is very rare. In such a case, each word could cost us O((n−k)⋅n⋅k)=O(n2⋅k−n⋅k2).
There are m words, which means calculating bold could cost O(m⋅(n2⋅k−n⋅k2)).
After calculating bold, we create the answer in O(n). This work is dominated by the other terms.
•	Space complexity: O(n)
We use the boolean array bold which has a length of n.

            */
            public string MarkBoldChars(string inputString, string[] words)
            {
                int stringLength = inputString.Length;
                bool[] isBold = new bool[stringLength];

                foreach (string word in words)
                {
                    int startIndex = inputString.IndexOf(word);
                    while (startIndex != -1)
                    {
                        for (int i = startIndex; i < startIndex + word.Length; i++)
                        {
                            isBold[i] = true;
                        }

                        startIndex = inputString.IndexOf(word, startIndex + 1);
                    }
                }

                string openingTag = "<b>";
                string closingTag = "</b>";
                System.Text.StringBuilder resultBuilder = new System.Text.StringBuilder();

                for (int i = 0; i < stringLength; i++)
                {
                    if (isBold[i] && (i == 0 || !isBold[i - 1]))
                    {
                        resultBuilder.Append(openingTag);
                    }

                    resultBuilder.Append(inputString[i]);

                    if (isBold[i] && (i == stringLength - 1 || !isBold[i + 1]))
                    {
                        resultBuilder.Append(closingTag);
                    }
                }

                return resultBuilder.ToString();
            }
        }


        /*
        604. Design Compressed String Iterator
        https://leetcode.com/problems/design-compressed-string-iterator/description/
        */
        public class StringIteratorSol
        {
            /*        
    Approach #1 Uncompressing the String [Time Limit Exceeded]
    Performance Analysis
    •	We precompute the elements of the uncompressed string. Thus, the space required in this case is O(m), where m refers to the length of the uncompressed string.
    •	The time required for precomputation is O(m) since we need to generate the uncompressed string of length m.
    •	Once the precomputation has been done, the time required for performing next() and hasNext() is O(1) for both.
    •	This approach can be easily extended to include previous(), last() and find() operations. All these operations require the use an index only and thus, take O(1) time. Operations like hasPrevious() can also be easily included.
    •	Since, once the precomputation has been done, next() requires O(1) time, this approach is useful if next() operation needs to be performed a large number of times. However, if hasNext() is performed most of the times, this approach isn't much advantageous since precomputation needs to be done anyhow.
    •	A potential problem with this approach could arise if the length of the uncompressed string is very large. In such a case, the size of the complete uncompressed string could become so large that it can't fit in the memory limits, leading to memory overflow.


            */
            public class StringIteratorWithUncompress
            {
                private StringBuilder resultStringBuilder = new StringBuilder();
                private int pointer = 0;

                public StringIteratorWithUncompress(string inputString)
                {
                    int index = 0;
                    while (index < inputString.Length)
                    {
                        char character = inputString[index++];
                        int number = 0;
                        while (index < inputString.Length && char.IsDigit(inputString[index]))
                        {
                            number = number * 10 + inputString[index] - '0';
                            index++;
                        }
                        for (int j = 0; j < number; j++)
                            resultStringBuilder.Append(character);
                    }
                }

                public char Next()
                {
                    if (!HasNext())
                        return ' ';
                    return resultStringBuilder[pointer++];
                }

                public bool HasNext()
                {
                    return pointer != resultStringBuilder.Length;
                }
            }

            /*
            Approach #2 Pre-Computation [Accepted]
         Performance Analysis
    •	The space required for storing the results of the precomputation is O(n), where n refers to the length of the compressed string. The nums and chars array contain a total of n elements.
    •	The precomputation step requires O(n) time. Thus, if hasNext() operation is performed most of the times, this precomputation turns out to be non-advantageous.
    •	Once the precomputation has been done, hasNext() and next() requires O(1) time.
    •	This approach can be extended to include the previous() and hasPrevious() operations, but that would require making some simple modifications to the current implementation.

            */
            public class StringIteratorWithPreCompute
            {
                private int currentPointer = 0;
                private string[] characterArray;
                private int[] numberArray;

                public StringIteratorWithPreCompute(string compressedString)
                {
                    numberArray = compressedString.Substring(1)
                        .Split(new[] { Regex.Escape("a-zA-Z") }, StringSplitOptions.RemoveEmptyEntries)
                        .Select(int.Parse)
                        .ToArray();
                    characterArray = Regex.Split(compressedString, "[0-9]+");
                }

                public char Next()
                {
                    if (!HasNext())
                        return ' ';
                    numberArray[currentPointer]--;
                    char result = characterArray[currentPointer][0];
                    if (numberArray[currentPointer] == 0)
                        currentPointer++;
                    return result;
                }

                public bool HasNext()
                {
                    return currentPointer != characterArray.Length - 1;
                }
            }
            /*
            Approach #3 Demand-Computation [Accepted]
            Performance Analysis**
    •	Since no precomputation is done, constant space is required in this case.
    •	The time required to perform next() operation is O(1).
    •	The time required for hasNext() operation is O(1).
    •	Since no precomputations are done, and hasNext() requires only O(1) time, this solution is advantageous if hasNext() operation is performed most of the times.
    •	This approach can be extended to include previous() and hasPrevious() operationsm, but this will require the use of some additional variables.

            */
            public class StringIteratorOnDemandCompute
            {
                private string result;
                private int pointer = 0;
                private int number = 0;
                private char character = ' ';

                public StringIteratorOnDemandCompute(string input)
                {
                    result = input;
                }

                public char Next()
                {
                    if (!HasNext())
                        return ' ';
                    if (number == 0)
                    {
                        character = result[pointer++];
                        while (pointer < result.Length && char.IsDigit(result[pointer]))
                        {
                            number = number * 10 + (result[pointer++] - '0');
                        }
                    }
                    number--;
                    return character;
                }

                public bool HasNext()
                {
                    return pointer != result.Length || number != 0;
                }
            }

        }

        /*
        443. String Compression
        https://leetcode.com/problems/string-compression/description/
        */

        public class StringCompressSol
        {
            /*         Complexity Analysis
Let n be the length of chars.
•	Time complexity: O(n).
All cells are initially white. We will repaint each white cell blue, and we may repaint some blue cells green. Thus each cell will be repainted at most twice. Since there are n cells, the total number of repaintings is O(n).
•	Space complexity: O(1).
We store only a few integer variables and the string representation of groupLength which takes up O(1) space. */
            public int Compress(char[] characters)
            {
                int index = 0, result = 0;
                while (index < characters.Length)
                {
                    int groupLength = 1;
                    while (index + groupLength < characters.Length && characters[index + groupLength] == characters[index])
                    {
                        groupLength++;
                    }
                    characters[result++] = characters[index];
                    if (groupLength > 1)
                    {
                        foreach (char digit in groupLength.ToString().ToCharArray())
                        {
                            characters[result++] = digit;
                        }
                    }
                    index += groupLength;
                }
                return result;
            }
        }

        /*     1531. String Compression II
        https://leetcode.com/problems/string-compression-ii/description/
         */
        public class GetLengthOfOptimalCompressionSol
        {
            private Dictionary<int, int> memo = new Dictionary<int, int>();
            private HashSet<int> add = new HashSet<int> { 1, 9, 99 };

            /* Approach 1: Dynamic Programming
            Complexity Analysis
            •	Time complexity: O(nk^2).
            Each recursive call will require only constant time, so we can calculate our time complexity as O(1) times the number of recursive calls made. Since we are using memoization, the number of recursive calls will be proportional to the number of DP states.
            Each DP state is defined as (idx, last, cnt, k), so we can calculate the number of DP states as the product of the number of possible values for each parameter. There will be n possible values for idx, A possible values for last, where A=26 is the size of the alphabet, n possible values for cnt, and k possible values for k. Thus, there are O(An^2k) possible DP states.
            However we can tighten our upper bound and get rid of A in our complexity. Let us look at pairs (last, cnt) and check how many of them we have. Consider the case aaababbbcc. Then for letter a we can have states (a, 1), (a, 2), (a, 3) and (a, 4), because we have only 4 letters a in our string and after deletions we can not have more. We have states (b, 1), (b, 2), (b, 3), (b, 4), (c, 1), (c, 2). Notice that some of these states can not be reached, because we do not have enough deletions. But what we know for sure is that the total number of pairs (last, cnt) is not more than n. Now we can adjust our analysis and we have:
            1.	When we consider our states, we have n options to choose index
            2.	We have n options in total to choose a pair (l, lc), because for each letter the maximum length is the frequency of this letter.
            3.	We have k+1 options to choose how many elements we need to delete: it can be (0, ..., k).
            Also we have at most 2 transitions from one state to another and it makes total time complexity O(nk^2).
            •	Space complexity: O(nk^2).
            We already found the number of states when we calculated time complexity, here is the same reasoning.

             */
            public int UsingDP(string s, int k)
            {
                return Dp(s, 0, (char)('a' + 26), 0, k);
            }

            private int Dp(string s, int idx, char lastChar, int lastCharCount, int k)
            {
                if (k < 0)
                {
                    return int.MaxValue / 2;
                }

                if (idx == s.Length)
                {
                    return 0;
                }

                int key = idx * 101 * 27 * 101 + (lastChar - 'a') * 101 * 101 + lastCharCount * 101 + k;

                if (memo.ContainsKey(key))
                {
                    return memo[key];
                }

                int keepChar;
                int deleteChar = Dp(s, idx + 1, lastChar, lastCharCount, k - 1);
                if (s[idx] == lastChar)
                {
                    keepChar = Dp(s, idx + 1, lastChar, lastCharCount + 1, k) + (add.Contains(lastCharCount) ? 1 : 0);
                }
                else
                {
                    keepChar = Dp(s, idx + 1, s[idx], 1, k) + 1;
                }
                int res = Math.Min(keepChar, deleteChar);
                memo[key] = res;

                return res;
            }
        }

        /*     3163. String Compression III
        https://leetcode.com/problems/string-compression-iii/description/
         */
        public class CompressedStringSol
        {
            /* Complexity
            Time complexity: O(n)

            Space complexity: O(1)

             */
            public string CompressedString(string word)
            {
                int n = word.Length, count = 0, i = 0, j = 0;
                StringBuilder ans = new StringBuilder();
                while (j < n)
                {
                    count = 0;
                    while (j < n && word[i] == word[j] && count < 9)
                    {
                        j++;
                        count++;
                    }
                    ans.Append(count).Append(word[i]);
                    i = j;
                }
                return ans.ToString();

            }
        }

        /*
        609. Find Duplicate File in System
https://leetcode.com/problems/find-duplicate-file-in-system/description/
        */
        public class FindDuplicateSol
        {
            /*
            Approach #1 Brute Force [Time Limit Exceeded]
            Complexity Analysis
            •	Time complexity : O(n∗x+f^2∗s). Creation of list will take O(n∗x), where n is the number of directories and x is the average string length. Every file is compared with every other file. Let f files are there with average size of s, then files comparision will take O(f^2∗s), equals can take O(s). Here, Worst case will be when all files are unique.
            •	Space complexity : O(n∗x). Size of lists res and list can grow upto n∗x.

            */
            public List<List<string>> Naive(string[] paths)
            {
                List<string[]> fileList = new List<string[]>();
                foreach (string path in paths)
                {
                    string[] values = path.Split(' ');
                    for (int i = 1; i < values.Length; i++)
                    {
                        string[] nameContent = values[i].Split('(');
                        nameContent[1] = nameContent[1].Replace(")", "");
                        fileList.Add(new string[] {
                    values[0] + "/" + nameContent[0], nameContent[1]
                });
                    }
                }
                bool[] visited = new bool[fileList.Count];
                List<List<string>> result = new List<List<string>>();
                for (int i = 0; i < fileList.Count - 1; i++)
                {
                    if (visited[i])
                        continue;
                    List<string> duplicateFiles = new List<string>();
                    for (int j = i + 1; j < fileList.Count; j++)
                    {
                        if (fileList[i][1].Equals(fileList[j][1]))
                        {
                            duplicateFiles.Add(fileList[j][0]);
                            visited[j] = true;
                        }
                    }
                    if (duplicateFiles.Count > 0)
                    {
                        duplicateFiles.Add(fileList[i][0]);
                        result.Add(duplicateFiles);
                    }
                }
                return result;
            }
            /*
            Approach #2 Using HashMap [Accepted]
            Complexity Analysis
            •	Time complexity : O(n∗x). n strings of average length x is parsed.
            •	Space complexity : O(n∗x). map and res size grows upto n∗x.

            */
            public List<List<string>> UsingHashMap(string[] paths)
            {
                Dictionary<string, List<string>> fileMap = new Dictionary<string, List<string>>();

                foreach (string path in paths)
                {
                    string[] values = path.Split(' ');
                    for (int i = 1; i < values.Length; i++)
                    {
                        string[] nameContent = values[i].Split('(');
                        nameContent[1] = nameContent[1].Replace(")", "");
                        List<string> fileList = fileMap.ContainsKey(nameContent[1]) ? fileMap[nameContent[1]] : new List<string>();
                        fileList.Add(values[0] + "/" + nameContent[0]);
                        fileMap[nameContent[1]] = fileList;
                    }
                }

                List<List<string>> result = new List<List<string>>();
                foreach (string key in fileMap.Keys)
                {
                    if (fileMap[key].Count > 1)
                        result.Add(fileMap[key]);
                }

                return result;
            }

        }
        /*
        621. Task Scheduler
    https://leetcode.com/problems/task-scheduler/description/
        */

        public class TaskSchedulerSol
        {
            /*
            Approach 1: Using Priority Queue / Max Heap
Complexity Analysis
Let the number of tasks be N. Let k be the size of the priority queue. k can, at maximum, be 26 because the priority queue stores the frequency of each distinct task, which is represented by the letters A to Z.
•	Time complexity: O(N)
In the worst case, all tasks must be processed, and each task might be inserted and extracted from the priority queue. The priority queue operations (insertion and extraction) have a time complexity of O(logk) each. Therefore, the overall time complexity is O(N⋅logk). Since k is at maximum 26, logk is a constant term. We can simplify the time complexity to O(N). This is a linear time complexity with a high constant factor.
•	Space complexity: O(26) = O(1)
The space complexity is mainly determined by the frequency array and the priority queue. The frequency array has a constant size of 26, and the priority queue can have a maximum size of 26 when all distinct tasks are present. Therefore, the overall space complexity is O(1) or O(26), which is considered constant.

            */
            public int UsingMaxHeap(char[] tasks, int n)
            {
                // Build frequency map
                int[] frequency = new int[26];
                foreach (char task in tasks)
                {
                    frequency[task - 'A']++;
                }

                // Max heap to store frequencies
                PriorityQueue<int, int> maxHeap = new PriorityQueue<int, int>();
                for (int i = 0; i < 26; i++)
                {
                    if (frequency[i] > 0)
                    {
                        maxHeap.Enqueue(frequency[i], frequency[i]);
                    }
                }

                int totalTime = 0;
                // Process tasks until the heap is empty
                while (maxHeap.Count > 0)
                {
                    int cycle = n + 1;
                    List<int> storedFrequencies = new List<int>();
                    int taskCount = 0;
                    // Execute tasks in each cycle
                    while (cycle-- > 0 && maxHeap.Count > 0)
                    {
                        int currentFrequency = maxHeap.Dequeue();
                        if (currentFrequency > 1)
                        {
                            storedFrequencies.Add(currentFrequency - 1);
                        }
                        taskCount++;
                    }
                    // Restore updated frequencies to the heap
                    foreach (int freq in storedFrequencies)
                    {
                        maxHeap.Enqueue(freq, freq);
                    }
                    // Add time for the completed cycle
                    totalTime += (maxHeap.Count == 0 ? taskCount : n + 1);
                }
                return totalTime;
            }

            /*
            Approach 2: Filling the Slots and Sorting
    Complexity Analysis
Let the number of tasks be N. There are up to 26 distinct tasks because the tasks are represented by the letters A to Z.
•	Time complexity: O(N)
The time complexity of the algorithm is O(26log26+N), where 26log26 is the time complexity of sorting the frequency array, and N is the length of the input task list, which is the dominating term.
•	Space complexity: O(26)=O(1)
The frequency array has a size of 26.
Note that some extra space is used when we sort arrays in place. The space complexity of the sorting algorithm depends on the programming language.
o	In Python, the sort method sorts a list using the Timsort algorithm which is a combination of Merge Sort and Insertion Sort and has O(N) additional space.
o	In Java, Arrays.sort() is implemented using a variant of the Quick Sort algorithm which has a space complexity of O(logN) for sorting two arrays.
o	In C++, the sort() function is implemented as a hybrid of Quick Sort, Heap Sort, and Insertion Sort, with a worse-case space complexity of O(logN).
We sort the frequency array, which has a size of 26. The space used for sorting takes O(26) or O(log26), which is constant, so the space complexity of the algorithm is O(26), which is constant, i.e. O(1).
        
            */
            public int FillSlotsAndSort(char[] tasks, int n)
            {
                // Create a frequency array to keep track of the count of each task
                int[] frequencyArray = new int[26];
                foreach (char task in tasks)
                {
                    frequencyArray[task - 'A']++;
                }

                // Sort the frequency array in non-decreasing order
                Array.Sort(frequencyArray);
                // Calculate the maximum frequency of any task
                int maximumFrequency = frequencyArray[25] - 1;
                // Calculate the number of idle slots that will be required
                int idleSlots = maximumFrequency * n;

                // Iterate over the frequency array from the second highest frequency to the lowest frequency
                for (int i = 24; i >= 0 && frequencyArray[i] > 0; i--)
                {
                    // Subtract the minimum of the maximum frequency and the current frequency from the idle slots
                    idleSlots -= Math.Min(maximumFrequency, frequencyArray[i]);
                }

                // If there are any idle slots left, add them to the total number of tasks
                return idleSlots > 0 ? idleSlots + tasks.Length : tasks.Length;
            }
            /*
            Approach 3: Greedy Approach
            Complexity Analysis
Let N be the number of tasks.
•	Time complexity: O(N)
To obtain count(A) and the count of tasks with the highest frequency, we iterate through the inputs, calculating counts for each distinct character. This process has a time complexity of O(N). All other operations have a time complexity of O(1), resulting in an overall time complexity of O(N)
•	Space complexity: O(26) = O(1)
The array count is size 26 because the tasks are represented by the letters A to Z. No data structures that vary with input size are used, resulting in an overall space complexity of O(1).

            */
            public int Greedy(char[] tasks, int n)
            {
                // Counter array to store the frequency of each task
                int[] taskFrequencyCounter = new int[26];
                int highestFrequency = 0;
                int highestFrequencyCount = 0;

                // Traverse through tasks to calculate task frequencies
                foreach (char task in tasks)
                {
                    taskFrequencyCounter[task - 'A']++;
                    if (highestFrequency == taskFrequencyCounter[task - 'A'])
                    {
                        highestFrequencyCount++;
                    }
                    else if (highestFrequency < taskFrequencyCounter[task - 'A'])
                    {
                        highestFrequency = taskFrequencyCounter[task - 'A'];
                        highestFrequencyCount = 1;
                    }
                }

                // Calculate idle slots, available tasks, and idles needed
                int totalParts = highestFrequency - 1;
                int partLength = n - (highestFrequencyCount - 1);
                int emptySlots = totalParts * partLength;
                int availableTasks = tasks.Length - highestFrequency * highestFrequencyCount;
                int additionalIdles = Math.Max(0, emptySlots - availableTasks);

                return tasks.Length + additionalIdles;
            }
            /*
Approach 4: Using Math Formula
Complexity Analysis
Let N be the number of tasks.
•	Time complexity: O(N)
The loop iterating over the tasks array has a time complexity of O(N). The loop iterating over the freq array has a time complexity proportional to the number of unique tasks, which is at most 26 because the tasks are represented by the letters A to Z. Therefore, the overall time complexity is O(N+26), which simplifies to O(N).
•	Space complexity: O(26) = O(1)
The freq array can store at most 26 unique tasks, resulting in O(26) space complexity. Other variables used in the algorithm have constant space requirements. Therefore, the overall space complexity is O(1).

            */
            public int UsingMathFormula(char[] tasks, int n)
            {
                // Frequency array to store the frequency of each task
                int[] frequencyArray = new int[26];
                int maximumFrequency = 0;

                // Count the frequency of each task and find the maximum frequency
                foreach (char task in tasks)
                {
                    frequencyArray[task - 'A']++;
                    maximumFrequency = Math.Max(maximumFrequency, frequencyArray[task - 'A']);
                }

                // Calculate the total time needed for execution
                int totalTime = (maximumFrequency - 1) * (n + 1);
                foreach (int frequency in frequencyArray)
                {
                    if (frequency == maximumFrequency)
                    {
                        totalTime++;
                    }
                }

                // Return the maximum of total time needed and the length of the task list
                return Math.Max(tasks.Length, totalTime);
            }

        }

        /*
        2365. Task Scheduler II	
        https://leetcode.com/problems/task-scheduler-ii/description/
        https://algo.monster/liteproblems/2365        
        */

        public class TaskSchedulerIISol
        {
            /*
            Approach1: HashMap
            Time and Space Complexity
The given Python function canMeasureWater determines whether it is possible to measure exactly targetCapacity liters by using two jugs of capacities jug1Capacity and jug2Capacity. It does so using a theorem related to the Diophantine equation which states that a target capacity x can be measured using two jugs with capacities m and n if and only if x is a multiple of the greatest common divisor (GCD) of m and n.
Time Complexity:
The time complexity of the function is predominantly determined by the computation of the GCD of jug1Capacity and jug2Capacity. Here’s how the complexity breaks down:
1.	The function checks if the sum of the capacities of the two jugs is less than the targetCapacity. This comparison is constant time, O(1).
2.	Then, it checks if either jug has a 0 capacity, and in such cases, it also performs constant-time comparisons: O(1).
3.	Finally, it calculates the GCD of the two jug capacities. The GCD is calculated using Euclid's algorithm, which has a worst-case time complexity of O(log(min(a, b))), where a and b are jug1Capacity and jug2Capacity. Since the GCD function is bounded by the smaller of the two numbers, the time complexity for this step is O(log(min(jug1Capacity, jug2Capacity))).
Therefore, the overall time complexity of the function is O(log(min(jug1Capacity, jug2Capacity))).
Space Complexity:
The space complexity of the function is determined by the space used to hold any variables and the stack space used by the recursion (if the implementation of GCD is recursive):
1.	Only a fixed number of integer variables are used, and there’s no use of any data structures that scale with the input size. This contributes a constant space complexity: O(1).
2.	Assuming gcd function from the math library is used, which is typically implemented iteratively, the space complexity remains constant as there are no recursive calls stacking up.
Therefore, the overall space complexity of the function is O(1) constant space.	
s
            */
            public long UsingHashMap(int[] tasks, int space)
            {
                // Map to store the next valid day a task can be scheduled
                Dictionary<int, long> nextValidDay = new Dictionary<int, long>();
                long currentDay = 0; // Represents the current day

                foreach (int task in tasks)         // Iterate through all tasks
                {
                    currentDay++; // Move to the next day
                                  // Check if we need to wait for a cooldown for the current task, and if necessary, jump to the next valid day
                    currentDay = Math.Max(currentDay, nextValidDay.GetValueOrDefault(task, 0L));
                    // Calculate and store the next valid day the current task can be executed based on the space (cooldown period)
                    nextValidDay[task] = currentDay + space + 1;

                }
                // The last value of currentDay will be the total time taken to complete all tasks
                return currentDay;

            }
            /*
Approach2: HashMap
Time O(n)
Space O(n)
*/
            public long UsingHashMap2(int[] tasks, int space)
            {
                Dictionary<int, long> next = new Dictionary<int, long>();
                long res = 0;
                foreach (int a in tasks)
                {
                    res = Math.Max(next.GetValueOrDefault(a, 0L), res + 1);
                    next[a] = res + space + 1;
                }
                return res;
            }
        }

        /*
        849. Maximize Distance to Closest Person	
        https://leetcode.com/problems/maximize-distance-to-closest-person/description/
        */
        public class MaxDistToClosestSol
        {
            /*
            Approach #1: Next Array [Accepted]
Complexity Analysis
•	Time Complexity: O(N), where N is the length of seats.
•	Space Complexity: O(N), the space used by left and right.

            */
            public int NextArray(int[] seats)
            {
                int numberOfSeats = seats.Length;
                int[] leftDistances = new int[numberOfSeats], rightDistances = new int[numberOfSeats];
                Array.Fill(leftDistances, numberOfSeats);
                Array.Fill(rightDistances, numberOfSeats);

                for (int index = 0; index < numberOfSeats; ++index)
                {
                    if (seats[index] == 1) leftDistances[index] = 0;
                    else if (index > 0) leftDistances[index] = leftDistances[index - 1] + 1;
                }

                for (int index = numberOfSeats - 1; index >= 0; --index)
                {
                    if (seats[index] == 1) rightDistances[index] = 0;
                    else if (index < numberOfSeats - 1) rightDistances[index] = rightDistances[index + 1] + 1;
                }

                int maximumDistance = 0;
                for (int index = 0; index < numberOfSeats; ++index)
                    if (seats[index] == 0)
                        maximumDistance = Math.Max(maximumDistance, Math.Min(leftDistances[index], rightDistances[index]));
                return maximumDistance;
            }

            /*            
Approach #2: Two Pointer [Accepted]
Complexity Analysis
•	Time Complexity: O(N), where N is the length of seats.
•	Space Complexity: O(1).

            */
            public int TwoPointer(int[] seats)
            {
                int N = seats.Length;
                int prev = -1, future = 0;
                int ans = 0;

                for (int i = 0; i < N; ++i)
                {
                    if (seats[i] == 1)
                    {
                        prev = i;
                    }
                    else
                    {
                        while (future < N && seats[future] == 0 || future < i)
                            future++;

                        int left = prev == -1 ? N : i - prev;
                        int right = future == N ? N : future - i;
                        ans = Math.Max(ans, Math.Min(left, right));
                    }
                }

                return ans;
            }
            /*
            Approach #3: Group by Zero [Accepted]
          Complexity Analysis
•	Time Complexity: O(N), where N is the length of seats.
•	Space Complexity: O(1).
  
            */
            public int maxDistToClosest(int[] seats)
            {
                int N = seats.Length;
                int K = 0; //current longest group of empty seats
                int ans = 0;

                for (int i = 0; i < N; ++i)
                {
                    if (seats[i] == 1)
                    {
                        K = 0;
                    }
                    else
                    {
                        K++;
                        ans = Math.Max(ans, (K + 1) / 2);
                    }
                }

                for (int i = 0; i < N; ++i) if (seats[i] == 1)
                    {
                        ans = Math.Max(ans, i);
                        break;
                    }

                for (int i = N - 1; i >= 0; --i) if (seats[i] == 1)
                    {
                        ans = Math.Max(ans, N - 1 - i);
                        break;
                    }

                return ans;
            }

        }

        /*
        640. Solve the Equation	
        https://leetcode.com/problems/solve-the-equation/editorial/
        */
        public class SolveEquationSol
        {
            /*
            Approach #1 Partioning Coefficients [Accepted]	
            Complexity Analysis
•	Time complexity : O(n). Generating coefficients and findinn lhs and rhs will take O(n).
•	Space complexity : O(n). ArrayList res size can grow upto n.

            */


            public string UsingCoefficientPartition(string equation)
            {
                string[] leftRight = equation.Split('=');
                int leftHandSide = 0, rightHandSide = 0;

                foreach (string term in BreakIt(leftRight[0]))
                {
                    if (term.IndexOf("x") >= 0)
                    {
                        leftHandSide += int.Parse(Coefficient(term));
                    }
                    else
                    {
                        rightHandSide -= int.Parse(term);
                    }
                }

                foreach (string term in BreakIt(leftRight[1]))
                {
                    if (term.IndexOf("x") >= 0)
                    {
                        leftHandSide -= int.Parse(Coefficient(term));
                    }
                    else
                    {
                        rightHandSide += int.Parse(term);
                    }
                }

                if (leftHandSide == 0)
                {
                    if (rightHandSide == 0)
                        return "Infinite solutions";
                    else
                        return "No solution";
                }
                return "x=" + (rightHandSide / leftHandSide).ToString();
            }
            public string Coefficient(string term)
            {
                if (term.Length > 1 && term[term.Length - 2] >= '0' && term[term.Length - 2] <= '9')
                    return term.Replace("x", "");
                return term.Replace("x", "1");
            }

            public List<string> BreakIt(string expression)
            {
                List<string> result = new List<string>();
                string currentTerm = "";

                for (int i = 0; i < expression.Length; i++)
                {
                    if (expression[i] == '+' || expression[i] == '-')
                    {
                        if (currentTerm.Length > 0)
                            result.Add(currentTerm);
                        currentTerm = "" + expression[i];
                    }
                    else
                    {
                        currentTerm += expression[i];
                    }
                }
                result.Add(currentTerm);
                return result;
            }
            /*
Approach #2 Using regex for spliting [Accepted]
Complexity Analysis
•	Time complexity : O(n). Generating coefficients and finding lhs and rhs will take O(n).
•	Space complexity : O(n). ArrayList res size can grow upto n.

            */
            public string RegExWithSplit(string equation)
            {
                string[] leftRight = equation.Split('=');
                int leftHandSide = 0, rightHandSide = 0;

                //TODO: Replace below with RegEx
                foreach (string term in leftRight[0].Split(new[] { '+', '-' }, StringSplitOptions.RemoveEmptyEntries))
                {
                    if (term.Contains("x"))
                    {
                        leftHandSide += int.Parse(Coefficient(term));
                    }
                    else
                    {
                        rightHandSide -= int.Parse(term);
                    }
                }

                //TODO: Replace below with RegEx

                foreach (string term in leftRight[1].Split(new[] { '+', '-' }, StringSplitOptions.RemoveEmptyEntries))
                {
                    if (term.Contains("x"))
                        leftHandSide -= int.Parse(Coefficient(term));
                    else
                        rightHandSide += int.Parse(term);
                }

                if (leftHandSide == 0)
                {
                    if (rightHandSide == 0)
                        return "Infinite solutions";
                    else
                        return "No solution";
                }
                else
                    return "x=" + (rightHandSide / leftHandSide);
            }

        }

        /*
        688. Knight Probability in Chessboard
        https://leetcode.com/problems/knight-probability-in-chessboard/description/
        */
        public class KnightProbabilitySol
        {
            /*
           Approach 1: Bottom-up Dynamic Programming
           Complexity Analysis
•	Time complexity: O(k⋅n^2).
We have four nested for-loops: for moves, for i, for j, and for direction. The outer loop for moves runs k times, the second and third loops for i and for j iterate over all cells on the n×n chessboard, and the innermost loop for direction iterates over the possible directions. As there are a constant number of directions (8), this loop can be considered as O(1) iterations.
Within each state (moves,i,j), the time complexity is constant, as we perform simple calculations and update the dynamic programming table.
The total number of iterations is determined by the product of the number of iterations in each loop: O(k⋅n^2).
•	Space complexity: O(k⋅n^2).
We use a three-dimensional dynamic programming table dp of size (k+1)×n×n to store the probabilities of being at each cell after a certain number of moves. Therefore, the space complexity is O(k⋅n^2).
 
            */
            public double BottomUpDP(int boardSize, int numberOfMoves, int startingRow, int startingColumn)
            {
                // Define possible directions for the knight's moves
                int[][] knightMoves = new int[][] {
            new int[] {1, 2}, new int[] {1, -2}, new int[] {-1, 2}, new int[] {-1, -2},
            new int[] {2, 1}, new int[] {2, -1}, new int[] {-2, 1}, new int[] {-2, -1}
        };

                // Initialize the dynamic programming table
                double[][][] probabilityTable = new double[numberOfMoves + 1][][];
                for (int i = 0; i <= numberOfMoves; i++)
                {
                    probabilityTable[i] = new double[boardSize][];
                    for (int j = 0; j < boardSize; j++)
                    {
                        probabilityTable[i][j] = new double[boardSize];
                    }
                }

                probabilityTable[0][startingRow][startingColumn] = 1.0;

                // Iterate over the number of moves
                for (int moves = 1; moves <= numberOfMoves; moves++)
                {
                    // Iterate over the cells on the chessboard
                    for (int currentRow = 0; currentRow < boardSize; currentRow++)
                    {
                        for (int currentColumn = 0; currentColumn < boardSize; currentColumn++)
                        {
                            // Iterate over possible directions
                            foreach (int[] move in knightMoves)
                            {
                                int previousRow = currentRow - move[0];
                                int previousColumn = currentColumn - move[1];
                                // Check if the previous cell is within the chessboard
                                if (previousRow >= 0 && previousRow < boardSize && previousColumn >= 0 && previousColumn < boardSize)
                                {
                                    // Add the previous probability divided by 8
                                    probabilityTable[moves][currentRow][currentColumn] += probabilityTable[moves - 1][previousRow][previousColumn] / 8.0;
                                }
                            }
                        }
                    }
                }

                // Calculate total probability by summing probabilities for all cells
                double totalProbability = 0.0;
                for (int i = 0; i < boardSize; i++)
                {
                    for (int j = 0; j < boardSize; j++)
                    {
                        totalProbability += probabilityTable[numberOfMoves][i][j];
                    }
                }
                return totalProbability;
            }

            /*
            Approach 2: Bottom-up Dynamic Programming with Optimized Space Complexity
          Complexity Analysis
•	Time complexity: O(k⋅n^2).
It is the same as in the previous approach.
•	Space complexity: O(n^2).
We use two dynamic programming tables: prev_dp and curr_dp, each of size n×n. Therefore, the space complexity is O(n^2). The space complexity does not depend on the number of moves k, as we only keep track of the probabilities of being at each cell after the previous and current moves.
  
            */
            public double BottomUpDPSpaceOptimal(int boardSize, int numberOfMoves, int startingRow, int startingColumn)
            {
                // Define possible directions for the knight's moves
                int[][] knightMoves = {
            new int[] {1, 2}, new int[] {1, -2}, new int[] {-1, 2}, new int[] {-1, -2},
            new int[] {2, 1}, new int[] {2, -1}, new int[] {-2, 1}, new int[] {-2, -1}
        };

                // Initialize the previous and current DP tables
                double[][] previousDP = new double[boardSize][];
                double[][] currentDP = new double[boardSize][];
                for (int i = 0; i < boardSize; i++)
                {
                    previousDP[i] = new double[boardSize];
                    currentDP[i] = new double[boardSize];
                }

                // Set the probability of the starting cell to 1
                previousDP[startingRow][startingColumn] = 1;

                // Iterate over the number of moves
                for (int moves = 1; moves <= numberOfMoves; moves++)
                {
                    // Iterate over the cells on the chessboard
                    for (int row = 0; row < boardSize; row++)
                    {
                        for (int column = 0; column < boardSize; column++)
                        {
                            currentDP[row][column] = 0;

                            // Iterate over possible directions
                            foreach (int[] move in knightMoves)
                            {
                                int previousRow = row - move[0];
                                int previousColumn = column - move[1];

                                // Check if the previous cell is within the chessboard
                                if (previousRow >= 0 && previousRow < boardSize && previousColumn >= 0 && previousColumn < boardSize)
                                {
                                    // Update the probability by adding the previous probability divided by 8
                                    currentDP[row][column] += previousDP[previousRow][previousColumn] / 8;
                                }
                            }
                        }
                    }

                    // Swap the previous and current DP tables
                    double[][] temp = previousDP;
                    previousDP = currentDP;
                    currentDP = temp;
                }

                // Calculate the total probability by summing up the probabilities for all cells
                double totalProbability = 0;
                for (int row = 0; row < boardSize; row++)
                {
                    for (int column = 0; column < boardSize; column++)
                    {
                        totalProbability += previousDP[row][column];
                    }
                }

                // Return the total probability
                return totalProbability;
            }
            /*
            Approach 3: Top-down Dynamic Programming (Memoization)
Complexity Analysis
•	Time complexity: O(k⋅n^2).
Even though we changed the order in which we calculate DP, the time complexity is the same as in the previous approach: for each state (moves,i,j), we calculate dp[moves][i][j] in O(1). Since we store the results in the memory, we will compute dp[moves][i][j] only once.
•	Space complexity: O(k⋅n^2).
We store the DP table of size [k+1][n][n].

            */
            int[][] directions = new int[][] { new int[] { 1, 2 }, new int[] { 1, -2 }, new int[] { -1, 2 }, new int[] { -1, -2 }, new int[] { 2, 1 }, new int[] { 2, -1 }, new int[] { -2, 1 }, new int[] { -2, -1 } };

            public double TopDownDPWithMemo(int n, int k, int row, int column)
            {
                double[][][] dp = new double[k + 1][][];
                for (int i = 0; i <= k; i++)
                {
                    dp[i] = new double[n][];
                    for (int j = 0; j < n; j++)
                    {
                        dp[i][j] = new double[n];
                        Array.Fill(dp[i][j], -1);
                    }
                }

                // Calculate the total probability by summing up the probabilities for all cells
                double totalProbability = 0;
                for (int i = 0; i < n; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        totalProbability += CalculateDP(dp, k, i, j, n, row, column);
                    }
                }

                return totalProbability;
            }
            private double CalculateDP(double[][][] dp, int moves, int i, int j, int n, int row, int column)
            {
                // Base case
                if (moves == 0)
                {
                    if (i == row && j == column)
                    {
                        return 1;
                    }
                    else
                    {
                        return 0;
                    }
                }

                // Check if the value has already been calculated
                if (dp[moves][i][j] != -1)
                {
                    return dp[moves][i][j];
                }

                dp[moves][i][j] = 0;

                // Iterate over possible directions
                foreach (int[] direction in directions)
                {
                    int prevI = i - direction[0];
                    int prevJ = j - direction[1];

                    // Boundary check
                    if (prevI >= 0 && prevI < n && prevJ >= 0 && prevJ < n)
                    {
                        dp[moves][i][j] += CalculateDP(dp, moves - 1, prevI, prevJ, n, row, column) / 8.0;
                    }
                }

                return dp[moves][i][j];
            }

        }


        /*
        690. Employee Importance
        https://leetcode.com/problems/employee-importance/description/	

        */
        class EmpImpSol
        {
            /*
            Approach #1: Depth-First Search [Accepted]
            Complexity Analysis
•	Time Complexity: O(N), where N is the number of employees. We might query each employee in dfs.
•	Space Complexity: O(N), the size of the implicit call stack when evaluating dfs.

            */
            private Dictionary<int, Employee> employeeMap;

            public int DFS(List<Employee> employees, int queryId)
            {
                employeeMap = new Dictionary<int, Employee>();
                foreach (Employee employee in employees)
                {
                    employeeMap[employee.Id] = employee;
                }
                return Dfs(queryId);
            }

            public int Dfs(int employeeId)
            {
                Employee employee = employeeMap[employeeId];
                int totalImportance = employee.Importance;
                foreach (int subordinateId in employee.Subordinates)
                {
                    totalImportance += Dfs(subordinateId);
                }
                return totalImportance;
            }
            public class Employee
            {
                public int Id;
                public int Importance;
                public IList<int> Subordinates;
            }
        }


        /*
        347. Top K Frequent Elements
        https://leetcode.com/problems/top-k-frequent-elements/description/ 
        */
        public class TopKFrequentElemSol
        {
            /*
            Approach 1: Heap
           Complexity Analysis
•	Time complexity : O(Nlogk) if k<N and O(N) in the particular case of N=k. That ensures time complexity to be better than O(NlogN).
•	Space complexity : O(N+k) to store the hash map with not more N elements and a heap with k elements.
 
            */
            public int[] UsingHeap(int[] numbers, int k)
            {
                // O(1) time
                if (k == numbers.Length)
                {
                    return numbers;
                }

                // 1. Build hash map: character and how often it appears
                // O(N) time
                Dictionary<int, int> frequencyCount = new Dictionary<int, int>();
                foreach (int number in numbers)
                {
                    if (frequencyCount.ContainsKey(number))
                    {
                        frequencyCount[number]++;
                    }
                    else
                    {
                        frequencyCount[number] = 1;
                    }
                }

                // init heap 'the less frequent element first'
                //TODO: find out type of heap (min/max?) below
                PriorityQueue<int, int> heap = new PriorityQueue<int, int>(
                    Comparer<int>.Create((n1, n2) => frequencyCount[n1].CompareTo(frequencyCount[n2])));

                // 2. Keep k top frequent elements in the heap
                // O(N log k) < O(N log N) time
                foreach (int number in frequencyCount.Keys)
                {
                    heap.Enqueue(number, number);
                    if (heap.Count > k) heap.Dequeue();
                }

                // 3. Build an output array
                // O(k log k) time
                int[] topKFrequent = new int[k];
                for (int i = k - 1; i >= 0; --i)
                {
                    topKFrequent[i] = heap.Dequeue();
                }
                return topKFrequent;
            }

            /*
            Approach 2: Quickselect (Hoare's selection algorithm)
           Complexity Analysis
•	Time complexity: O(N) in the average case,
O(N2) in the worst case. Please refer to this card for a good detailed explanation of Master Theorem. Master Theorem helps to get an average complexity by writing the algorithm cost as T(N)=aT(N/b)+f(N). Here we have an example of Master Theorem case III: T(N)=T(2N)+N, which results in O(N) time complexity. That's the case with random pivots.
In the worst case of constantly badly chosen pivots, the problem is not divided by half at each step, it becomes just one element less, which leads to O(N2) time complexity. It happens, for example, if at each step you choose the pivot not randomly, but take the rightmost element. For the random pivot choice, the probability of having such a worst-case is negligibly small.
•	Space complexity: up to O(N) to store hash map and array of unique elements.
 
            */
            private int[] unique;
            private Dictionary<int, int> count;
            public int[] Quickselect(int[] nums, int k)
            {
                // Build hash map: character and how often it appears
                count = new Dictionary<int, int>();
                foreach (int num in nums)
                {
                    if (count.ContainsKey(num))
                        count[num]++;
                    else
                        count[num] = 1;
                }

                // Array of unique elements
                int n = count.Count;
                unique = new int[n];
                int index = 0;
                foreach (int num in count.Keys)
                {
                    unique[index] = num;
                    index++;
                }

                // kth top frequent element is (n - k)th less frequent.
                // Do a partial sort: from less frequent to the most frequent, till
                // (n - k)th less frequent element takes its place (n - k) in a sorted array. 
                // All elements on the left are less frequent.
                // All the elements on the right are more frequent. 
                QuickSelectAlgo(0, n - 1, n - k);
                // Return top k frequent elements
                int[] result = new int[k];
                Array.Copy(unique, n - k, result, 0, k);
                return result;
            }
            public void Swap(int firstIndex, int secondIndex)
            {
                int temporaryValue = unique[firstIndex];
                unique[firstIndex] = unique[secondIndex];
                unique[secondIndex] = temporaryValue;
            }

            public int Partition(int left, int right, int pivotIndex)
            {
                int pivotFrequency = count[unique[pivotIndex]];
                // 1. Move pivot to end
                Swap(pivotIndex, right);
                int storeIndex = left;

                // 2. Move all less frequent elements to the left
                for (int i = left; i <= right; i++)
                {
                    if (count[unique[i]] < pivotFrequency)
                    {
                        Swap(storeIndex, i);
                        storeIndex++;
                    }
                }

                // 3. Move the pivot to its final place
                Swap(storeIndex, right);

                return storeIndex;
            }

            public void QuickSelectAlgo(int left, int right, int kSmallest)
            {
                /*
                Sort a list within left..right till kth less frequent element
                takes its place. 
                */

                // base case: the list contains only one element
                if (left == right) return;

                // Select a random pivotIndex
                Random randomNum = new Random();
                int pivotIndex = left + randomNum.Next(right - left + 1);

                // Find the pivot position in a sorted list
                pivotIndex = Partition(left, right, pivotIndex);

                // If the pivot is in its final sorted position
                if (kSmallest == pivotIndex)
                {
                    return;
                }
                else if (kSmallest < pivotIndex)
                {
                    // go left
                    QuickSelectAlgo(left, pivotIndex - 1, kSmallest);
                }
                else
                {
                    // go right 
                    QuickSelectAlgo(pivotIndex + 1, right, kSmallest);
                }
            }


        }
        /*
        692. Top K Frequent Words
        https://leetcode.com/problems/top-k-frequent-words/description/

        */
        public class TopKFrequentWordsSol
        {
            /*            
Approach 1: Brute Force
Complexity Analysis
let N be the length of words.
•	Time Complexity: O(NlogN). We count the frequency of each word in O(N) time, and then we sort the given words in O(NlogN) time.
•	Space Complexity: O(N), the space used to store frequencies in a HashMap and return a slice from a sorted list of length O(N).

            */
            public List<string> Naive(string[] words, int k)
            {
                Dictionary<string, int> wordCount = new Dictionary<string, int>();
                foreach (string word in words)
                {
                    if (wordCount.ContainsKey(word))
                    {
                        wordCount[word]++;
                    }
                    else
                    {
                        wordCount[word] = 1;
                    }
                }

                List<string> candidates = new List<string>(wordCount.Keys);
                candidates.Sort((word1, word2) =>
                    wordCount[word1] == wordCount[word2] ? word1.CompareTo(word2) : wordCount[word2] - wordCount[word1]);

                return candidates.GetRange(0, k);
            }

            /*
            Approach 2: PriorityQueue - Max Heap 
Complexity Analysis
Let N be the length of words.
•	Time Complexity: O(N+klogN). We count the frequency of each word in O(N) time and then heapify the list of unique words in O(N) time. Each time we pop the top from the heap, it costs logN time as the size of the heap is O(N).
•	Space Complexity: O(N), the space used to store our counter cnt and heap h.

            */
            public List<string> MaxHeapPQ(string[] words, int k)
            {
                Dictionary<string, int> countMap = new Dictionary<string, int>();
                foreach (string word in words)
                {
                    if (countMap.ContainsKey(word))
                    {
                        countMap[word]++;
                    }
                    else
                    {
                        countMap[word] = 1;
                    }
                }

                List<Word> candidates = new List<Word>();
                foreach (var entry in countMap)
                {
                    candidates.Add(new Word(entry.Key, entry.Value));
                }

                // Use a priority queue to get the top k frequent words
                candidates.Sort();
                List<string> result = new List<string>();
                for (int i = 0; i < k && i < candidates.Count; i++)
                {
                    result.Add(candidates[i].word);
                }
                return result;
            }
            public class Word : IComparable<Word>
            {
                public string word;
                private int count;

                public Word(string word, int count)
                {
                    this.word = word;
                    this.count = count;
                }

                public int CompareTo(Word other)
                {
                    if (this.count == other.count)
                    {
                        return this.word.CompareTo(other.word);
                    }
                    return other.count - this.count;
                }
            }
            /*
Approach 3: Min Heap
Complexity Analysis
•	Time Complexity: O(Nlogk), where N is the length of words. We count the frequency of each word in O(N) time, then we add N words to the heap, each in O(logk) time. Finally, we pop from the heap up to k times or just sort all elements in the heap as the returned result, which takes O(klogk). As k≤N, O(N)+O(Nlogk)+O(klogk)=O(Nlogk)
•	Space Complexity: O(N), O(N) space is used to store our counter cnt while O(k) space is for the heap.

            */
            public IList<string> MinHeapPQ(string[] words, int k)
            {
                Dictionary<string, int> wordCount = new Dictionary<string, int>();
                foreach (string word in words)
                {
                    if (wordCount.ContainsKey(word))
                    {
                        wordCount[word]++;
                    }
                    else
                    {
                        wordCount[word] = 1;
                    }
                }

                // Custom comparator for priority queue
                //TODO: Replace below SortedSet with Priority quque - Min Heap
                var priorityQueue = new SortedSet<string>(Comparer<string>.Create((w1, w2) =>
                {
                    int countComparison = wordCount[w1].CompareTo(wordCount[w2]);
                    return countComparison == 0 ? w2.CompareTo(w1) : countComparison;
                }));

                foreach (string word in wordCount.Keys)
                {
                    priorityQueue.Add(word);
                    if (priorityQueue.Count > k)
                    {
                        priorityQueue.Remove(priorityQueue.Min);
                    }
                }

                List<string> result = new List<string>();
                while (priorityQueue.Count > 0)
                {
                    result.Add(priorityQueue.Max);
                    priorityQueue.Remove(priorityQueue.Max);
                }

                result.Reverse();
                return result;
            }

            /*
            Approach 4: Bucket Sorting + Trie
            Complexity Analysis
Let N be the length of words.
•	Time Complexity: O(N). We take O(N) time to count frequencies and enumerate all buckets. Since we only need to get k words from tries, we traverse k paths in tries, and each path is neglectable in length (≤10), O(k) time is required to generate all those words from tries. Besides, it takes O(N) time to put N words in tries. As k≤N, O(N+k)=O(N)
•	Space Complexity: O(N), like other approaches, our counter cnt needs O(N) space. Besides, tries to store at most N words also needs O(n) space.
Note: Though we optimize the time complexity to O(N), it may run slower than previous approaches due to the large constant factors.

            */
            private int topK;
            private List<string> result;

            private class TrieNode
            {
                public TrieNode[] Children;
                public bool IsWord;

                public TrieNode()
                {
                    Children = new TrieNode[26];
                    IsWord = false;
                }
            }

            public IList<string> BucketSortWithTrie(string[] words, int k)
            {
                this.topK = k;
                result = new List<string>();
                int n = words.Length;
                TrieNode[] bucket = new TrieNode[n + 1];
                Dictionary<string, int> count = new Dictionary<string, int>();

                foreach (string word in words)
                {
                    if (count.ContainsKey(word))
                    {
                        count[word]++;
                    }
                    else
                    {
                        count[word] = 1;
                    }
                }

                foreach (var entry in count)
                {
                    if (bucket[entry.Value] == null)
                    {
                        bucket[entry.Value] = new TrieNode();
                    }
                    AddWord(bucket[entry.Value], entry.Key);
                }

                for (int i = n; i > 0; i--)
                {
                    if (bucket[i] != null)
                    {
                        GetWords(bucket[i], "");
                    }
                    if (this.topK == 0)
                    {
                        break;
                    }
                }
                return result;
            }

            private void AddWord(TrieNode root, string word)
            {
                TrieNode cur = root;
                foreach (char c in word)
                {
                    if (cur.Children[c - 'a'] == null)
                    {
                        cur.Children[c - 'a'] = new TrieNode();
                    }
                    cur = cur.Children[c - 'a'];
                }
                cur.IsWord = true;
            }

            private void GetWords(TrieNode root, string prefix)
            {
                if (topK == 0)
                {
                    return;
                }
                if (root.IsWord)
                {
                    topK--;
                    result.Add(prefix);
                }
                for (int i = 0; i < 26; i++)
                {
                    if (root.Children[i] != null)
                    {
                        GetWords(root.Children[i], prefix + (char)(i + 'a'));
                    }
                }
            }


        }


        /*
        973. K Closest Points to Origin
        https://leetcode.com/problems/k-closest-points-to-origin/description/
        */
        public class KClosestPointsToOriginSol
        {
            /*
            Approach 1: Sort with Custom Comparator
            Complexity Analysis
Here N refers to the length of the given array points.
•	Time complexity: O(N⋅logN) for the sorting of points.
While sorting methods vary between different languages, most have a worst-case or average time complexity of O(N⋅logN).
•	Space complexity: O(logN) to O(N) for the extra space required by the sorting process.
As with the time complexity, the space complexity of the sorting method used can vary from language to language. C++'s STL, for example, uses QuickSort most of the time but will switch to either HeapSort or InsertionSort depending on the nature of the data. Java uses a variant of QuickSort with dual pivots when dealing with arrays of primitive values. The implementation of both C++'s and Java's sort methods will require an average of O(logN) extra space. Python, on the other hand, uses TimSort, which is a hybrid of MergeSort and InsertionSort and requires O(N) extra space. Unlike most other languages, Javascript's sort method will actually vary from browser to browser. Since the adoption of ECMAScript 2019, however, the sort method is required to be stable, which generally means MergeSort or TimSort and a space complexity of O(N).

            */
            public int[][] SortWithCustomCompare(int[][] points, int k)
            {
                // Sort the array with a custom comparison
                Array.Sort(points, (a, b) => SquaredDistance(a) - SquaredDistance(b));

                // Return the first k elements of the sorted array
                return points.Take(k).ToArray();
            }

            private int SquaredDistance(int[] point)
            {
                // Calculate and return the squared Euclidean distance
                return point[0] * point[0] + point[1] * point[1];
            }

            /*
            Approach 2: Max Heap or Max Priority Queue
         Complexity Analysis
Here N refers to the length of the given array points.
•	Time complexity: O(N⋅logk)
Adding to/removing from the heap (or priority queue) only takes O(logk) time when the size of the heap is capped at k elements.
•	Space complexity: O(k)
The heap (or priority queue) will contain at most k elements.
   
            */
            public int[][] MaxHeapPQ(int[][] points, int k)
            {
                // Use a lambda comparator to sort the PQ by farthest distance
                PriorityQueue<int[], int[]> maxPriorityQueue = new PriorityQueue<int[], int[]>(Comparer<int[]>.Create((a, b) => b[0].CompareTo(a[0])));
                for (int i = 0; i < points.Length; i++)
                {
                    int[] entry = { SquaredDistance(points[i]), i };
                    if (maxPriorityQueue.Count < k)
                    {
                        // Fill the max PQ up to k points
                        maxPriorityQueue.Enqueue(entry, entry);
                    }
                    else if (entry[0] < maxPriorityQueue.Peek()[0])
                    {
                        // If the max PQ is full and a closer point is found,
                        // discard the farthest point and add this one
                        maxPriorityQueue.Dequeue();
                        maxPriorityQueue.Enqueue(entry, entry);
                    }
                }

                // Return all points stored in the max PQ
                int[][] answer = new int[k][];
                for (int i = 0; i < k; i++)
                {
                    int entryIndex = maxPriorityQueue.Dequeue()[1];
                    answer[i] = points[entryIndex];
                }
                return answer;
            }
            /*
            Approach 3: Binary Search
            Complexity Analysis
Here N refers to the length of the given array points.
•	Time complexity: O(N)
While this binary search variant has a worst-case time complexity of O(N2), it has an average time complexity of O(N). It achieves this by halving (on average) the remaining elements needing to be processed at each iteration, which results in N+N/2+N/4+N/8+...+N/N=2N total processes. This yields an average time complexity of O(N).
•	Space complexity: O(N)
An extra O(N) space is required for the arrays containing distances and reference indices.	

            */
            public int[][] UsingBinarySearch(int[][] points, int k)
            {
                // Precompute the Euclidean distance for each point,
                // define the initial binary search range,
                // and create a reference list of point indices
                double[] distances = new double[points.Length];
                double low = 0, high = 0;
                List<int> remainingPoints = new List<int>();
                for (int i = 0; i < points.Length; i++)
                {
                    distances[i] = EuclideanDistance(points[i]);
                    high = Math.Max(high, distances[i]);
                    remainingPoints.Add(i);
                }

                // Perform a binary search of the distances
                // to find the k closest points
                List<int> closestPoints = new List<int>();
                while (k > 0)
                {
                    double mid = low + (high - low) / 2;
                    List<List<int>> result = SplitDistances(remainingPoints, distances, mid);
                    List<int> closer = result[0], farther = result[1];
                    if (closer.Count > k)
                    {
                        // If more than k points are in the closer distances
                        // then discard the farther points and continue
                        remainingPoints = closer;
                        high = mid;
                    }
                    else
                    {
                        // Add the closer points to the answer array and keep
                        // searching the farther distances for the remaining points
                        k -= closer.Count;
                        closestPoints.AddRange(closer);
                        remainingPoints = farther;
                        low = mid;
                    }
                }

                // Return the k closest points using the reference indices
                k = closestPoints.Count;
                int[][] answer = new int[k][];
                for (int i = 0; i < k; i++)
                {
                    answer[i] = points[closestPoints[i]];
                }
                return answer;
            }

            private List<List<int>> SplitDistances(List<int> remainingPoints, double[] distances, double mid)
            {
                // Split the distances around the midpoint
                // and return them in separate lists
                List<List<int>> result = new List<List<int>> {
            new List<int>(),
            new List<int>()
        };
                foreach (int point in remainingPoints)
                {
                    if (distances[point] <= mid)
                    {
                        result[0].Add(point);
                    }
                    else
                    {
                        result[1].Add(point);
                    }
                }
                return result;
            }

            private double EuclideanDistance(int[] point)
            {
                // Calculate and return the squared Euclidean distance
                return point[0] * point[0] + point[1] * point[1];
            }

            /*
            Approach 4: QuickSelect
Complexity Analysis
Here N refers to the length of the given array points.
•	Time complexity: O(N).
Similar to the earlier binary search solution, the QuickSelect solution has a worst-case time complexity of O(N^2) if the worst pivot is chosen each time. On average, however, it has a time complexity of O(N) because it halves (roughly) the remaining elements needing to be processed at each iteration. This results in N+N/2+N/4+N/8+...+N/N=2N total processes, yielding an average time complexity of O(N).
•	Space complexity: O(1).
The QuickSelect algorithm conducts the partial sort of points in place with no recursion, so only constant extra space is required.

            */
            public int[][] QuickSelect(int[][] points, int k)
            {
                return QuickSelectAlgo(points, k);
            }

            private int[][] QuickSelectAlgo(int[][] points, int k)
            {
                int left = 0, right = points.Length - 1;
                int pivotIndex = points.Length;
                while (pivotIndex != k)
                {
                    // Repeatedly partition the array
                    // while narrowing in on the kth element
                    pivotIndex = Partition(points, left, right);
                    if (pivotIndex < k)
                    {
                        left = pivotIndex;
                    }
                    else
                    {
                        right = pivotIndex - 1;
                    }
                }

                // Return the first k elements of the partially sorted array
                int[][] result = new int[k][];
                Array.Copy(points, result, k);
                return result;
            }

            private int Partition(int[][] points, int left, int right)
            {
                int[] pivot = ChoosePivot(points, left, right);
                int pivotDistance = SquaredDistance(pivot);
                while (left < right)
                {
                    // Iterate through the range and swap elements to make sure
                    // that all points closer than the pivot are to the left
                    if (SquaredDistance(points[left]) >= pivotDistance)
                    {
                        int[] temp = points[left];
                        points[left] = points[right];
                        points[right] = temp;
                        right--;
                    }
                    else
                    {
                        left++;
                    }
                }

                // Ensure the left pointer is just past the end of
                // the left range then return it as the new pivotIndex
                if (SquaredDistance(points[left]) < pivotDistance)
                    left++;
                return left;
            }

            private int[] ChoosePivot(int[][] points, int left, int right)
            {
                // Choose a pivot element of the array
                return points[left + (right - left) / 2];
            }
        };

        /*
        1772. Sort Features by Popularity
        https://leetcode.com/problems/sort-features-by-popularity/description/
        */
        public class SortFeaturesSol
        {
            /*
            Approach1: Hash Map + Sort
           Time : n = Features.Length : m = response.length ,    k = response.length() : longest respose String .
 T = O(nlogn ) for sorting + O(m.k) ==> for find a word in response String and update frequency .
 Split function's time complexity = O(k) 
 so, if we  assume  k ~= log n , n ~= m , Then   T = O(n log n).
 
 Space = O(k)==> HashSet/ Split words used to store words     
            */
            public string[] HashMapWithSort(string[] features, string[] responses)
            {
                Dictionary<string, int> frequencyMap = new Dictionary<string, int>();

                // Initialize frequency for features 
                foreach (string feature in features)
                {
                    frequencyMap[feature] = 0;
                }

                foreach (string response in responses)
                {
                    // Split response string and put in HashSet - to remove duplicates.
                    HashSet<string> uniqueWordsSet = new HashSet<string>(response.Split(' '));

                    // If any word in set is a key in frequency map, then update frequency for that word.
                    foreach (string word in uniqueWordsSet)
                    {
                        if (frequencyMap.ContainsKey(word))
                        {
                            frequencyMap[word]++; // Increment frequency 
                        }
                    }
                }

                // Sort features as per frequency: decreasing order: highest frequency to lowest 
                Array.Sort(features, (a, b) => frequencyMap[b].CompareTo(frequencyMap[a]));
                return features;
            }
        }


        /*
        2284. Sender With Largest Word Count
        https://leetcode.com/problems/sender-with-largest-word-count/description/ 	
        */
        public class SenderWithLargestWordCount
        {
            /*
            Method 1: Sort the senders with the largest word count
Perf Analysis:
Time: O(nlogn), space: O(n), where n = senders.length.
            */
            public string WithSorting(string[] messages, string[] senders)
            {
                Dictionary<string, int> senderMessageCount = new Dictionary<string, int>();
                int largestWordCount = 0;

                for (int i = 0; i < senders.Length; ++i)
                {
                    int wordCount = messages[i].Split(' ').Length;
                    if (senderMessageCount.ContainsKey(senders[i]))
                    {
                        senderMessageCount[senders[i]] += wordCount;
                    }
                    else
                    {
                        senderMessageCount[senders[i]] = wordCount;
                    }
                    largestWordCount = Math.Max(largestWordCount, senderMessageCount[senders[i]]);
                }

                SortedSet<string> senderSet = new SortedSet<string>();
                foreach (var entry in senderMessageCount)
                {
                    if (entry.Value == largestWordCount)
                    {
                        senderSet.Add(entry.Key);
                    }
                }

                return senderSet.Last();
            }
            /*
            Method 2: No sort
Perf Analysis:
Time & space: O(n), where n = senders.length.
            */
            public string WithOutSorting(string[] messages, string[] senders)
            {
                Dictionary<string, int> messageCount = new Dictionary<string, int>();
                int largestCount = 0;
                for (int i = 0; i < senders.Length; ++i)
                {
                    int wordCount = messages[i].Split(' ').Length;
                    if (messageCount.ContainsKey(senders[i]))
                    {
                        messageCount[senders[i]] += wordCount;
                    }
                    else
                    {
                        messageCount[senders[i]] = wordCount;
                    }
                    largestCount = Math.Max(largestCount, messageCount[senders[i]]);
                }
                string senderWithLargestCount = "";
                foreach (var entry in messageCount)
                {
                    if (entry.Value == largestCount && string.Compare(senderWithLargestCount, entry.Key) < 0)
                    {
                        senderWithLargestCount = entry.Key;
                    }
                }
                return senderWithLargestCount;
            }

        }

        /*
        699. Falling Squares
        https://leetcode.com/problems/falling-squares/description/ 

        */
        class FallingSquaresSol
        {
            /*
Approach 1: Offline Propagation
Complexity Analysis
•	Time Complexity: O(N^2), where N is the length of positions. We use two for-loops, each of complexity O(N).
•	Space Complexity: O(N), the space used by qans and ans.

            */
            public List<int> OfflinePropagation(int[][] positions)
            {
                int[] squareHeights = new int[positions.Length];
                for (int i = 0; i < positions.Length; i++)
                {
                    int left = positions[i][0];
                    int size = positions[i][1];
                    int right = left + size;
                    squareHeights[i] += size;

                    for (int j = i + 1; j < positions.Length; j++)
                    {
                        int left2 = positions[j][0];
                        int size2 = positions[j][1];
                        int right2 = left2 + size2;
                        if (left2 < right && left < right2)
                        { //intersect
                            squareHeights[j] = Math.Max(squareHeights[j], squareHeights[i]);
                        }
                    }
                }

                List<int> result = new List<int>();
                int currentMaxHeight = -1;
                foreach (int height in squareHeights)
                {
                    currentMaxHeight = Math.Max(currentMaxHeight, height);
                    result.Add(currentMaxHeight);
                }
                return result;
            }
            /*
            
Approach 2: Brute Force with Coordinate Compression
Complexity Analysis
•	Time Complexity: O(N^2), where N is the length of positions. We use two for-loops, each of complexity O(N) (because of coordinate compression.)
•	Space Complexity: O(N), the space used by heights.

            */
            public List<int> NaiveWithCoordinateCompression(int[][] positions)
            {
                // Coordinate Compression
                HashSet<int> coords = new HashSet<int>();
                foreach (int[] pos in positions)
                {
                    coords.Add(pos[0]);
                    coords.Add(pos[0] + pos[1] - 1);
                }
                List<int> sortedCoords = new List<int>(coords);
                sortedCoords.Sort();

                Dictionary<int, int> index = new Dictionary<int, int>();
                int t = 0;
                foreach (int coord in sortedCoords)
                {
                    index.Add(coord, t++);
                }
                int bestHeight = 0;
                List<int> answerList = new List<int>();

                foreach (var pos in positions)
                {
                    int left = index[pos[0]];
                    int right = index[pos[0] + pos[1] - 1];
                    int height = Query(left, right) + pos[1];
                    Update(left, right, height);
                    bestHeight = Math.Max(bestHeight, height);
                    answerList.Add(bestHeight);
                }
                return answerList;
            }
            int[] heights;

            public int Query(int left, int right)
            {
                int answer = 0;
                for (int i = left; i <= right; i++)
                {
                    answer = Math.Max(answer, heights[i]);
                }
                return answer;
            }
            public void Update(int left, int right, int height)
            {
                for (int i = left; i <= right; i++)
                {
                    heights[i] = Math.Max(heights[i], height);
                }
            }


            /*
            Approach 3: Block (Square Root) Decomposition
Complexity Analysis
•	Time Complexity: O(N*Sqrt of N), where N is the length of positions. Each query and update has complexity O(Sqrt of N).
•	Space Complexity: O(N), the space used by heights.

            */
            private int[] blocks;
            private int[] blocksRead;
            private int blockSize;
            public List<int> BlockDemComposition(int[][] positions)
            {
                // Coordinate Compression
                HashSet<int> coords = new HashSet<int>();
                foreach (int[] pos in positions)
                {
                    coords.Add(pos[0]);
                    coords.Add(pos[0] + pos[1] - 1);
                }
                List<int> sortedCoords = new List<int>(coords);
                sortedCoords.Sort();

                Dictionary<int, int> index = new Dictionary<int, int>();
                int t = 0;
                foreach (int coord in sortedCoords)
                {
                    index.Add(coord, t++);
                }

                heights = new int[t];
                blockSize = (int)Math.Sqrt(t);
                blocks = new int[blockSize + 2];
                blocksRead = new int[blockSize + 2];

                int bestHeight = 0;
                List<int> result = new List<int>();

                foreach (int[] position in positions)
                {
                    int left = index[position[0]];
                    int right = index[position[0] + position[1] - 1];
                    int height = Query(left, right) + position[1];
                    Update(left, right, height);
                    bestHeight = Math.Max(bestHeight, height);
                    result.Add(bestHeight);
                }
                return result;

                int Query(int left, int right)
                {
                    int answer = 0;
                    while (left % blockSize > 0 && left <= right)
                    {
                        answer = Math.Max(answer, heights[left]);
                        answer = Math.Max(answer, blocks[left / blockSize]);
                        left++;
                    }
                    while (right % blockSize != blockSize - 1 && left <= right)
                    {
                        answer = Math.Max(answer, heights[right]);
                        answer = Math.Max(answer, blocks[right / blockSize]);
                        right--;
                    }
                    while (left <= right)
                    {
                        answer = Math.Max(answer, blocks[left / blockSize]);
                        answer = Math.Max(answer, blocksRead[left / blockSize]);
                        left += blockSize;
                    }
                    return answer;
                }

                void Update(int left, int right, int height)
                {
                    while (left % blockSize > 0 && left <= right)
                    {
                        heights[left] = Math.Max(heights[left], height);
                        blocksRead[left / blockSize] = Math.Max(blocksRead[left / blockSize], height);
                        left++;
                    }
                    while (right % blockSize != blockSize - 1 && left <= right)
                    {
                        heights[right] = Math.Max(heights[right], height);
                        blocksRead[right / blockSize] = Math.Max(blocksRead[right / blockSize], height);
                        right--;
                    }
                    while (left <= right)
                    {
                        blocks[left / blockSize] = Math.Max(blocks[left / blockSize], height);
                        left += blockSize;
                    }
                }
            }

            /*
            Approach 4: Segment Tree with Lazy Propagation
            Complexity Analysis
            •	Time Complexity: O(NlogN), where N is the length of positions. This is the run-time complexity of using a segment tree.
            •	Space Complexity: O(N), the space used by our tree.

            */
            public List<int> SegmentTreeWithLazyPropagation(int[][] positions)
            {
                // Coordinate Compression
                HashSet<int> coords = new HashSet<int>();
                foreach (int[] pos in positions)
                {
                    coords.Add(pos[0]);
                    coords.Add(pos[0] + pos[1] - 1);
                }
                List<int> sortedCoords = new List<int>(coords);
                sortedCoords.Sort();

                Dictionary<int, int> index = new Dictionary<int, int>();
                int t = 0;
                foreach (int coord in sortedCoords)
                {
                    index.Add(coord, t++);
                }

                SegmentTree segmentTree = new SegmentTree(sortedCoords.Count);
                int highestHeight = 0;
                List<int> results = new List<int>();

                foreach (int[] position in positions)
                {
                    int leftIndex = index[position[0]];
                    int rightIndex = index[position[0] + position[1] - 1];
                    int height = segmentTree.Query(leftIndex, rightIndex) + position[1];
                    segmentTree.Update(leftIndex, rightIndex, height);
                    highestHeight = Math.Max(highestHeight, height);
                    results.Add(highestHeight);
                }
                return results;
            }
            class SegmentTree
            {
                int size, height;
                int[] tree, lazy;

                public SegmentTree(int size)
                {
                    this.size = size;
                    height = 1;
                    while ((1 << height) < size)
                    {
                        height++;
                    }
                    tree = new int[2 * size];
                    lazy = new int[size];
                }

                private void Apply(int index, int value)
                {
                    tree[index] = Math.Max(tree[index], value);
                    if (index < size)
                    {
                        lazy[index] = Math.Max(lazy[index], value);
                    }
                }

                private void Pull(int index)
                {
                    while (index > 1)
                    {
                        index >>= 1;
                        tree[index] = Math.Max(tree[index * 2], tree[index * 2 + 1]);
                        tree[index] = Math.Max(tree[index], lazy[index]);
                    }
                }

                private void Push(int index)
                {
                    for (int h = height; h > 0; h--)
                    {
                        int y = index >> h;
                        if (lazy[y] > 0)
                        {
                            Apply(y * 2, lazy[y]);
                            Apply(y * 2 + 1, lazy[y]);
                            lazy[y] = 0;
                        }
                    }
                }

                public void Update(int left, int right, int height)
                {
                    left += size;
                    right += size;
                    int left0 = left, right0 = right;

                    while (left <= right)
                    {
                        if ((left & 1) == 1)
                        {
                            Apply(left++, height);
                        }
                        if ((right & 1) == 0)
                        {
                            Apply(right--, height);
                        }
                        left >>= 1;
                        right >>= 1;
                    }
                    Pull(left0);
                    Pull(right0);
                }

                public int Query(int left, int right)
                {
                    left += size;
                    right += size;
                    int result = 0;
                    Push(left);
                    Push(right);
                    while (left <= right)
                    {
                        if ((left & 1) == 1)
                        {
                            result = Math.Max(result, tree[left++]);
                        }
                        if ((right & 1) == 0)
                        {
                            result = Math.Max(result, tree[right--]);
                        }
                        left >>= 1;
                        right >>= 1;
                    }
                    return result;
                }


            }
        }

        /*
        218. The Skyline Problem	
        https://leetcode.com/problems/the-skyline-problem/description/ 
        */
        public class GetSkylineSol
        {
            /*
            
Approach 1: Brute Force I

            Complexity Analysis
Let n be the length of the input array buildings.
•	Time complexity: O(n^2)
o	Obtaining our sorted list of positions will require an average of O(nlogn) time.
o	Then for each of the n buildings, we need to update the maximum heights at all the indexes covered by its left edge and right edge. In the worst-case scenario, we have to update n values in each iteration step, so this process will take O(n^2) time.
•	Space complexity: O(n)
o	The number of left and right edges is 2n, thus we need a set and an array of size O(n).
o	Then we need a hash table of indexes and an array of heights, both of size O(n).
o	We also use an answer array to store all the skyline points, of which there are at most n.

            */
            public IList<IList<int>> Naive1(int[][] buildings)
            {
                // Sort the unique positions of all the edges.
                SortedSet<int> edgeSet = new SortedSet<int>();
                foreach (int[] building in buildings)
                {
                    int left = building[0], right = building[1];
                    edgeSet.Add(left);
                    edgeSet.Add(right);
                }
                List<int> edges = new List<int>(edgeSet);

                // Hash table 'edgeIndexMap' record every {position : index} pairs in edges.
                Dictionary<int, int> edgeIndexMap = new Dictionary<int, int>();
                for (int i = 0; i < edges.Count; ++i)
                {
                    edgeIndexMap[edges[i]] = i;
                }

                // Initialize 'heights' to record maximum height at each index.
                int[] heights = new int[edges.Count];

                // Iterate over all the buildings.
                foreach (int[] building in buildings)
                {
                    // For each building, find the indexes of its left
                    // and right edges.
                    int left = building[0], right = building[1], height = building[2];
                    int leftIndex = edgeIndexMap[left], rightIndex = edgeIndexMap[right];

                    // Update the maximum height within the range [leftIndex, rightIndex)
                    for (int idx = leftIndex; idx < rightIndex; ++idx)
                    {
                        heights[idx] = Math.Max(heights[idx], height);
                    }
                }

                List<IList<int>> answer = new List<IList<int>>();

                // Iterate over 'heights'.
                for (int i = 0; i < heights.Length; ++i)
                {
                    int currHeight = heights[i], currPos = edges[i];

                    // Add all the positions where the height changes to 'answer'.
                    if (answer.Count == 0 || (int)answer[answer.Count - 1][1] != currHeight)
                    {
                        answer.Add(new List<int> { currPos, currHeight });
                    }
                }
                return answer;
            }

            /*
            Approach 2: Brute Force II, Sweep Line
            Complexity Analysis
            Let n be the length of the input array buildings.
            •	Time complexity: O(n^2)
            o	Obtaining our sorted list of positions will require an average of O(nlogn) time.
            o	Then for each of the 2n positions we need to check if any of the n buildings intersect with the line at that position. This process will take O(n^2) time.
            •	Space complexity: O(n)
            o	The number of left and right edges is 2n, thus we need a set and an array of size O(n).
            o	We also use an answer array to store all the skyline points, of which there are at most n.

            */
            public IList<IList<int>> Naive2WithSweepLine(int[][] buildings)
            {
                // Collect and sort the unique positions of all the edges.
                SortedSet<int> edgeSet = new SortedSet<int>();
                int maxHeight, left, right, height;
                foreach (var building in buildings)
                {
                    left = building[0];
                    right = building[1];
                    edgeSet.Add(left);
                    edgeSet.Add(right);
                }
                List<int> positions = new List<int>(edgeSet);
                positions.Sort();

                // 'answer' for skyline key points.
                List<IList<int>> answer = new List<IList<int>>();


                // For each position, draw an imaginary vertical line.
                foreach (int position in positions)
                {
                    // The current max height.
                    maxHeight = 0;

                    // Iterate over all the buildings:
                    foreach (var building in buildings)
                    {
                        left = building[0];
                        right = building[1];
                        height = building[2];

                        // If the current building intersects with the line,
                        // update 'maxHeight'.
                        if (left <= position && position < right)
                        {
                            maxHeight = Math.Max(maxHeight, height);
                        }
                    }

                    // If it's the first key point or the height changes,
                    // we add [position, maxHeight] to 'answer'.
                    if (answer.Count == 0 || (int)answer[answer.Count - 1][1] != maxHeight)
                    {
                        answer.Add(new List<int> { position, maxHeight });
                    }
                }

                // Return 'answer' as the skyline.
                return answer;
            }

            /*
            Approach 3: Sweep Line + Priority Queue
            Complexity Analysis
            Let n be the length of the input array buildings.
            •	Time complexity: O(nlogn)
            o	There are 2n edges so we have at most O(n) unique positions during the iteration.
            o	At each step, we need to pop out the passed buildings from priority queue live and put in the newly added building (if exist). In worse-case scenario, we have O(n) live buildings in live, both the pop and push operations take O(logn) time.
            o	To sum up, the overall time complexity is O(nlogn).
            •	Space complexity: O(n)
            o	We initalize edges of size O(2n) to store all the edges and its indexes, empty list answer to store all the skyline key points.
            o	We maintain a priority queue live which has at most O(n) elements.
            o	There can be at most O(n) skyline key points, thus answer takes at most O(n) space.
            o	Therefore, the overall space complexity is O(n).

            */
            public IList<IList<int>> SweepLineWithPQ(int[][] buildings)
            {
                // Iterate over all buildings, for each building i
                // add (position, i) to edges.
                List<List<int>> edges = new List<List<int>>();
                for (int i = 0; i < buildings.Length; ++i)
                {
                    edges.Add(new List<int> { buildings[i][0], i });
                    edges.Add(new List<int> { buildings[i][1], i });
                }
                edges.Sort((a, b) => a[0].CompareTo(b[0]));

                // Initialize an empty Priority Queue 'live' to store all the newly 
                // added buildings, an empty list answer to store the skyline key points.
                //TODO: Replace below SortedSet with PriorityQueue
                SortedSet<List<int>> live = new SortedSet<List<int>>(Comparer<List<int>>.Create((a, b) =>
                {
                    int heightComparison = b[0].CompareTo(a[0]);
                    return heightComparison != 0 ? heightComparison : a[1].CompareTo(b[1]);
                }));
                IList<IList<int>> answer = new List<IList<int>>();

                int idx = 0;

                // Iterate over all the sorted edges.
                while (idx < edges.Count)
                {
                    // Since we might have multiple edges at same x,
                    // Let the 'currX' be the current position.
                    int currX = edges[idx][0];

                    // While we are handling the edges at 'currX':
                    while (idx < edges.Count && edges[idx][0] == currX)
                    {
                        // The index 'b' of this building in 'buildings'
                        int b = edges[idx][1];

                        // If this is a left edge of building 'b', we
                        // add (height, right) of building 'b' to 'live'.
                        if (buildings[b][0] == currX)
                        {
                            int right = buildings[b][1];
                            int height = buildings[b][2];
                            live.Add(new List<int> { height, right });
                        }
                        idx += 1;
                    }

                    // If the tallest live building has been passed,
                    // we remove it from 'live'.
                    while (live.Count > 0 && live.First()[1] <= currX)
                        live.Remove(live.First());

                    // Get the maximum height from 'live'.
                    int currHeight = live.Count == 0 ? 0 : live.First()[0];

                    // If the height changes at this currX, we add this
                    // skyline key point [currX, max_height] to 'answer'.
                    if (answer.Count == 0 || answer[answer.Count - 1][1] != currHeight)
                        answer.Add(new List<int> { currX, currHeight });
                }

                // Return 'answer' as the skyline.
                return answer;
            }
            /*
            Approach 4: Sweep Line + Two Priority Queue
            Complexity Analysis
            Let n be the length of the input array buildings.
            •	Time complexity: O(n⋅logn)
            o	We sort a list with length of 2⋅n, which takes O(n) time.
            o	Then we iterate over all the sorted edges, during the iteration, we have to manipulate on two priority queues, the amortized cost of this operation is O(logn).
            o	To sum up, the overall time complexity is O(n⋅logn)
            •	Space complexity: O(n)
            o	We used an empty array edges to store the information of all the left and right edges. There are 2⋅n edges and will cost O(n) space.
            o	Besides, we need to maintain two priority queues, in the worst-case scenario, each of them takes O(n) space.
            o	To sum up, the overall space complexity is O(n).

            */
            public IList<IList<int>> SweepLineWith2PQ(int[][] buildings)
            {
                // Iterate over all buildings, for each building = [left, right, height]
                // add (left, height) and (right, height) to 'edges'.
                List<List<int>> edges = new List<List<int>>();
                for (int i = 0; i < buildings.Length; ++i)
                {
                    edges.Add(new List<int> { buildings[i][0], buildings[i][2] });
                    edges.Add(new List<int> { buildings[i][1], -buildings[i][2] });
                }
                edges.Sort((a, b) => a[0].CompareTo(b[0]));

                // Initialize two empty priority queues 'live' and 'past',
                // an empty list 'answer' to store the skyline key points.
                PriorityQueue<int, int> live = new PriorityQueue<int, int>(Comparer<int>.Create((a, b) => b.CompareTo(a)));
                PriorityQueue<int, int> past = new PriorityQueue<int, int>(Comparer<int>.Create((a, b) => b.CompareTo(a)));
                IList<IList<int>> answer = new List<IList<int>>();

                int idx = 0;

                // Iterate over all the sorted edges.
                while (idx < edges.Count)
                {
                    // Since we might have multiple edges at same x,
                    // Let the 'currX' be the current position.
                    int currX = edges[idx][0];

                    // While we are handling the edges at 'currX':
                    while (idx < edges.Count && edges[idx][0] == currX)
                    {
                        // The height of the current building.
                        int height = edges[idx][1];

                        // If this is a left edge, add `height` to 'live'.
                        // Otherwise, add `height` to `past`.
                        if (height > 0)
                        {
                            live.Enqueue(height, height);
                        }
                        else
                        {
                            past.Enqueue(height, -height); //TODO: Double check this Enqueue
                        }
                        idx++;
                    }

                    // If the tallest live building has been passed,
                    // we remove it from 'live'.
                    while (past.Count > 0 && live.Count > 0 && live.Peek() == past.Peek())
                    {
                        live.Dequeue();
                        past.Dequeue();
                    }

                    // Get the maximum height from 'live'.
                    int currHeight = live.Count == 0 ? 0 : live.Peek();

                    // If the height changes at 'currX', we add this
                    // skyline key point [currX, max_height] to 'answer'.
                    if (answer.Count == 0 || answer[answer.Count - 1][1] != currHeight)
                    {
                        answer.Add(new List<int> { currX, currHeight });
                    }
                }

                // Return 'answer' as the skyline.
                return answer;
            }
            /*
            Approach 5: Union Find
            Complexity Analysis
            Let n be the length of the input array buildings.
            •	Time complexity: O(nlogn)
            o	Sorting the n buildings has an average time complexity of O(nlogn), though sorting algorithms vary by language.
            o	There are at most 2n unique positions for 2n edges, and sorting them similarly has an average time complexity of O(nlogn).
            o	The UnionFind.union() function has a time complexity of O(1) and will run at most 2n times for an overall time complexity of O(n).
            o	The UnionFind.find() function has a time complexity of O(n) for the worst-case scenario, but using a collapsing find technique brings this down to O(1) with repeated use. This amortizes to an overall time complexity of O(n), as each successful find() will update a value in root, and there are up to 2n elements in root. As shown in the picture below.

            •	Space complexity: O(n)
            o	There are at most 2n edges, thus the set edgeSet, the lists edges, heights, and answers, the union-find's root list, and the recursion stack for the union-find's find() are each limited to O(n) space.

            */
            class UnionFind
            {
                private int[] root;

                public UnionFind(int n)
                {
                    this.root = new int[n];
                    for (int i = 0; i < n; ++i)
                        root[i] = i;
                }

                public int Find(int x)
                {
                    return root[x] == x ? x : (root[x] = Find(root[x]));
                }

                public void Union(int x, int y)
                {
                    root[x] = root[y];
                }
            }

            public IList<IList<int>> WithUnionFind(int[][] buildings)
            {
                // Sort the unique positions of all the edges.
                SortedSet<int> edgeSet = new SortedSet<int>();
                foreach (int[] building in buildings)
                {
                    edgeSet.Add(building[0]);
                    edgeSet.Add(building[1]);
                }
                int[] edges = new List<int>(edgeSet).ToArray();
                Array.Sort(edges);

                // Hash table 'edgeIndexMap' record every {position : index} pairs in edges.
                Dictionary<int, int> edgeIndexMap = new Dictionary<int, int>();
                for (int i = 0; i < edges.Length; ++i)
                    edgeIndexMap[edges[i]] = i;

                // Sort buildings by descending order of heights.
                Array.Sort(buildings, (a, b) => b[2] - a[2]);

                // Initialize a disjoint set for all indexes, each index's
                // root is itself. Since there is no building added yet,
                // the height at each position is 0.
                int n = edges.Length;
                UnionFind edgeUF = new UnionFind(n);
                int[] heights = new int[n];

                // Iterate over all the buildings by descending height.
                foreach (int[] building in buildings)
                {
                    int leftEdge = building[0], rightEdge = building[1];
                    int height = building[2];

                    // For current x position, get the corresponding index.
                    int leftIndex = edgeIndexMap[leftEdge], rightIndex = edgeIndexMap[rightEdge];

                    // While we haven't update the the root of 'leftIndex':
                    while (leftIndex < rightIndex)
                    {
                        // Find the root of leftIndex, that is:
                        // The rightmost index having the same height as 'leftIndex'.
                        leftIndex = edgeUF.Find(leftIndex);

                        // If leftIndex < rightIndex, we have to update both the root and height
                        // of 'leftIndex', and move on to the next index towards 'rightIndex'.
                        // That is: increment 'leftIndex' by 1.
                        if (leftIndex < rightIndex)
                        {
                            edgeUF.Union(leftIndex, rightIndex);
                            heights[leftIndex] = height;
                            leftIndex++;
                        }
                    }
                }

                // Finally, we just need to iterate over updated heights, and
                // add every skyline key point to 'answer'.
                List<IList<int>> answer = new List<IList<int>>();
                for (int i = 0; i < n; ++i)
                {
                    if (i == 0 || heights[i] != heights[i - 1])
                        answer.Add(new List<int> { edges[i], heights[i] });
                }
                return answer;
            }
            /*
            Approach 6: Divide-and-Conquer
            Complexity Analysis
            Let n be the length of the input array buildings.
            •	Time complexity: O(nlogn)
            o	During the divide-and-conquer process, we recursively cut the array into two halves, thus logn steps are needed to split the original input array into single buildings and then merge them back together. In other words, the recursion stack has a depth of logn levels.
            o	In each level of the recursion, it takes a total of O(n) time to merge all the sub-skylines into larger skylines.
            •	Space complexity: O(n)
            o	We need O(n) space to create the answer array to record the merged skylines as there are at most 2n skyline key points.
            o	The recursion stack also requires an additional O(logn) space.


            */
            public IList<IList<int>> DivideAndConquer(int[][] buildings)
            {
                // Get the whole skyline from all the input buildings.
                return DivideAndConquer(buildings, 0, buildings.Length - 1);
            }

            public IList<IList<int>> DivideAndConquer(int[][] buildings, int left, int right)
            {
                // If the given array of building contains only 1 building, we can
                // directly return the corresponding skyline.
                if (left == right)
                {
                    IList<IList<int>> answer = new List<IList<int>>();
                    answer.Add(new List<int> { buildings[left][0], buildings[left][2] });
                    answer.Add(new List<int> { buildings[left][1], 0 });
                    return answer;
                }

                // Otherwise, we shall recursively divide the buildings and 
                // merge the skylines. Cut the given skyline into two halves, 
                // get skyline from each half and merge them into a single skyline.
                int mid = (right - left) / 2 + left;
                IList<IList<int>> leftSkyline = DivideAndConquer(buildings, left, mid);
                IList<IList<int>> rightSkyline = DivideAndConquer(buildings, mid + 1, right);

                return MergeSkylines(leftSkyline, rightSkyline);
            }
            public IList<IList<int>> MergeSkylines(IList<IList<int>> leftSkyline, IList<IList<int>> rightSkyline)
            {
                IList<IList<int>> answer = new List<IList<int>>();
                int leftPos = 0, rightPos = 0;
                int leftPrevHeight = 0, rightPrevHeight = 0;
                int curX, curY;

                while (leftPos < leftSkyline.Count && rightPos < rightSkyline.Count)
                {
                    int nextLeftX = leftSkyline[leftPos][0];
                    int nextRightX = rightSkyline[rightPos][0];

                    if (nextLeftX < nextRightX)
                    {
                        leftPrevHeight = leftSkyline[leftPos][1];
                        curX = nextLeftX;
                        curY = Math.Max(leftPrevHeight, rightPrevHeight);
                        leftPos++;
                    }
                    else if (nextLeftX > nextRightX)
                    {
                        rightPrevHeight = rightSkyline[rightPos][1];
                        curX = nextRightX;
                        curY = Math.Max(leftPrevHeight, rightPrevHeight);
                        rightPos++;
                    }
                    else
                    {
                        leftPrevHeight = leftSkyline[leftPos][1];
                        rightPrevHeight = rightSkyline[rightPos][1];
                        curX = nextLeftX;
                        curY = Math.Max(leftPrevHeight, rightPrevHeight);
                        leftPos++;
                        rightPos++;
                    }

                    if (answer.Count == 0 || answer[answer.Count - 1][1] != curY)
                    {
                        answer.Add(new List<int> { curX, curY });
                    }
                }

                while (leftPos < leftSkyline.Count)
                {
                    answer.Add(leftSkyline[leftPos]);
                    leftPos++;
                }

                while (rightPos < rightSkyline.Count)
                {
                    answer.Add(rightSkyline[rightPos]);
                    rightPos++;
                }

                return answer;
            }


        }

        /*
        703. Kth Largest Element in a Stream
        https://leetcode.com/problems/kth-largest-element-in-a-stream/description/
        */
        public class KthLargestInStreamSol
        {

            /*

    Approach 1: Maintain Sorted List
    Complexity Analysis
    Let M be the size of the initial stream nums given in the constructor. Let N be the number of calls of add.
    •	Time Complexity: O(N^2+N⋅M)
    The constructor involves creating a list stream from nums, which takes O(M) time. Then, sorting this list takes O(M⋅logM) time. Thus, the time complexity of the constructor is O(M⋅logM) time.
    The add function involves running a binary search on stream. Because the total size of stream at the end would be O(M+N), each binary search is bounded by a time complexity of O(log(M+N)). Moreover, adding a number in stream can take worst-case O(M+N) time, as adding an element in the middle of a list can offset all the elements to its right. Then, the time complexity of a single add call would be O(M+N+log(M+N)). Because add is called N times, the time complexity of all the add calls would be O(N⋅(M+N+log(M+N))).
    We see that after expanding the time complexity for the add function, the N⋅M and N2 terms dominate all the other log terms in our calculations, so the total time complexity is O(N^2+N⋅M)
    •	Space Complexity: O(M+N)
    The maximum size for stream is M+N, so the total space complexity is O(M+N).

            */
            public class WithSortedList
            {
                private List<int> numberStream;
                private int k;

                public WithSortedList(int k, int[] numbers)
                {
                    numberStream = new List<int>(numbers.Length);
                    this.k = k;

                    foreach (int number in numbers)
                    {
                        numberStream.Add(number);
                    }

                    numberStream.Sort();
                }

                public int Add(int value)
                {
                    int index = GetIndex(value);
                    // Add value to correct position
                    numberStream.Insert(index, value);
                    return numberStream[numberStream.Count - k];
                }

                private int GetIndex(int value)
                {
                    int left = 0;
                    int right = numberStream.Count - 1;
                    while (left <= right)
                    {
                        int mid = (left + right) / 2;
                        int midElement = numberStream[mid];
                        if (midElement == value) return mid;
                        if (midElement > value)
                        {
                            // Go to left half
                            right = mid - 1;
                        }
                        else
                        {
                            // Go to right half
                            left = mid + 1;
                        }
                    }
                    return left;
                }
            }

            /*
            Approach 2: Heap
            Complexity Analysis
            Let M be the size of the initial stream nums given in the constructor, and let N be the number of calls to add.
            •	Time Complexity: O((M+N)⋅logk)
            The add function involves adding and removing an element from a heap of size k, which is an O(logk) operation. Since the add function is called N times, the total time complexity for all add calls is O(N⋅logk).
            The constructor also calls add M times to initialize the heap, leading to a time complexity of O(M⋅logk).
            Therefore, the overall time complexity is O((M+N)⋅logk).
            •	Space Complexity: O(k)
            The minHeap maintains at most k elements, so the space complexity is O(k).	

            */
            class WithMinHeap
            {
                private PriorityQueue<int, int> minHeap;
                private int k;

                public WithMinHeap(int k, int[] nums)
                {
                    minHeap = new PriorityQueue<int, int>();
                    this.k = k;

                    foreach (int number in nums)
                    {
                        Add(number);
                    }
                }

                public int Add(int value)
                {
                    // Add to our minHeap if we haven't processed k elements yet
                    // or if value is greater than the top element (the k-th largest)
                    if (minHeap.Count < k || minHeap.Peek() < value)
                    {
                        minHeap.Enqueue(value, value);
                        if (minHeap.Count > k)
                        {
                            minHeap.Dequeue();
                        }
                    }
                    return minHeap.Peek();
                }
            }

        }
        /*
        710. Random Pick with Blacklist
        https://leetcode.com/problems/random-pick-with-blacklist/description/ 
        */
        class RandomPickWithBlackListSol
        {

            /*
            Approach: HashMap
            Perf Analysis: 
            O(B) / O(1), 

            */
            // N: [0, N)
            // B: blacklist
            // B1: < N
            // B2: >= N
            // M: N - B1
            int M;
            Random r;
            Dictionary<int, int> map;

            public RandomPickWithBlackListSol(int N, int[] blacklist)
            {
                map = new Dictionary<int, int>();
                foreach (int b in blacklist) // O(B)
                    map.Add(b, -1);
                M = N - map.Count;

                foreach (int b in blacklist)
                { // O(B)
                    if (b < M)
                    { // re-mapping
                        while (map.ContainsKey(N - 1))
                            N--;
                        map[b] = N - 1;
                        N--;
                    }
                }

                r = new Random();
            }

            public int Pick()
            {
                int p = r.Next(M);
                if (map.ContainsKey(p))
                    return map[p];
                return p;
            }
        }

        /*
        723. Candy Crush
        https://leetcode.com/problems/candy-crush/description/	
        */

        public class CandyCrushSol
        {
            int rowCount, columnCount;


            /*
            Approach 1: Separate Steps: Find, Crush, Drop
            Complexity Analysis
            Let m×n be the size of the grid board.
            •	Time complexity: O(m^2⋅n^2)
            o	Each find process takes O(m⋅n) time as we need to iterate over every cell of board.
            o	There could be at most O(m⋅n) independent drop steps to eliminate all valid candy groups, as shown in the picture below:

            We can construct the following board where around half of the candies ((m⋅n)/2) are crushed, and each crush operation eliminates at most two groups (8) of candies. Therefore, we need at least (m⋅n)/16 drops to obtain the final board.
            o	In summary, the time complexity in the worst-case scenario is O(m^2⋅n^2).


            •	Space complexity: O(m⋅n)
            o	In each find step, we store the crushable candies in crushed_set, there can be at most O(m⋅n) candies in the set (imagine all candies are of the same value).
            o	The drop and crush steps involve in-place modification and do not require additional space.

            */
            public int[,] WithSeperateSteps(int[,] board)
            {
                rowCount = board.GetLength(0);
                columnCount = board.GetLength(1);
                HashSet<Tuple<int, int>> crushedSet = Find(board);
                while (crushedSet.Count > 0)
                {
                    Crush(board, crushedSet);
                    Drop(board);
                    crushedSet = Find(board);
                }

                return board;

                HashSet<Tuple<int, int>> Find(int[,] board)
                {
                    HashSet<Tuple<int, int>> crushedSet = new HashSet<Tuple<int, int>>();

                    // Check vertically adjacent candies
                    for (int row = 1; row < rowCount - 1; row++)
                    {
                        for (int column = 0; column < columnCount; column++)
                        {
                            if (board[row, column] == 0)
                            {
                                continue;
                            }
                            if (board[row, column] == board[row - 1, column] && board[row, column] == board[row + 1, column])
                            {
                                crushedSet.Add(Tuple.Create(row, column));
                                crushedSet.Add(Tuple.Create(row - 1, column));
                                crushedSet.Add(Tuple.Create(row + 1, column));
                            }
                        }
                    }

                    // Check horizontally adjacent candies
                    for (int row = 0; row < rowCount; row++)
                    {
                        for (int column = 1; column < columnCount - 1; column++)
                        {
                            if (board[row, column] == 0)
                            {
                                continue;
                            }
                            if (board[row, column] == board[row, column - 1] && board[row, column] == board[row, column + 1])
                            {
                                crushedSet.Add(Tuple.Create(row, column));
                                crushedSet.Add(Tuple.Create(row, column - 1));
                                crushedSet.Add(Tuple.Create(row, column + 1));
                            }
                        }
                    }
                    return crushedSet;
                }

                void Crush(int[,] board, HashSet<Tuple<int, int>> crushedSet)
                {
                    foreach (Tuple<int, int> pair in crushedSet)
                    {
                        int row = pair.Item1;
                        int column = pair.Item2;
                        board[row, column] = 0;
                    }
                }

                void Drop(int[,] board)
                {
                    for (int column = 0; column < columnCount; column++)
                    {
                        int lowestZeroRow = -1;

                        // Iterate over each column
                        for (int row = rowCount - 1; row >= 0; row--)
                        {
                            if (board[row, column] == 0)
                            {
                                lowestZeroRow = Math.Max(lowestZeroRow, row);
                            }
                            else if (lowestZeroRow >= 0)
                            {
                                int temp = board[row, column];
                                board[row, column] = board[lowestZeroRow, column];
                                board[lowestZeroRow, column] = temp;
                                lowestZeroRow--;
                            }
                        }
                    }
                }
            }

            /*
        Approach 2: In-place Modification 
        Complexity Analysis
Let m×n be the size of the grid board.
•	Time complexity: O(m^2⋅n^2)
o	Each find_and_crush process takes O(m⋅n) time as we need to iterate over every cell of board.
o	There could be at most O(m⋅n) independent drop steps to eliminate all valid candy groups.
o	In summary, the time complexity in the worst-case scenario is O(m^2⋅n^2).
•	Space complexity: O(1)
o	Both the function drop and find_and_crush involve in-place modification and do not require additional space.
   
            */
            int rows, columns;
            public int[][] WithInPlaceModification(int[][] board)
            {
                rows = board.Length;
                columns = board[0].Length;

                // Continue with the three steps until we can no longer find any crushable candies.
                while (!FindAndCrush(board))
                {
                    Drop(board);
                }

                return board;
            }

            bool FindAndCrush(int[][] board)
            {
                bool isComplete = true;

                // Check vertically adjacent candies
                for (int row = 1; row < rows - 1; row++)
                {
                    for (int column = 0; column < columns; column++)
                    {
                        if (board[row][column] == 0)
                        {
                            continue;
                        }
                        if (Math.Abs(board[row][column]) == Math.Abs(board[row - 1][column]) && Math.Abs(board[row][column]) == Math.Abs(board[row + 1][column]))
                        {
                            board[row][column] = -Math.Abs(board[row][column]);
                            board[row - 1][column] = -Math.Abs(board[row - 1][column]);
                            board[row + 1][column] = -Math.Abs(board[row + 1][column]);
                            isComplete = false;
                        }
                    }
                }

                // Check horizontally adjacent candies
                for (int row = 0; row < rows; row++)
                {
                    for (int column = 1; column < columns - 1; column++)
                    {
                        if (board[row][column] == 0)
                        {
                            continue;
                        }
                        if (Math.Abs(board[row][column]) == Math.Abs(board[row][column - 1]) && Math.Abs(board[row][column]) == Math.Abs(board[row][column + 1]))
                        {
                            board[row][column] = -Math.Abs(board[row][column]);
                            board[row][column - 1] = -Math.Abs(board[row][column - 1]);
                            board[row][column + 1] = -Math.Abs(board[row][column + 1]);
                            isComplete = false;
                        }
                    }
                }

                // Set the value of each candy to be crushed as 0
                for (int row = 0; row < rows; row++)
                {
                    for (int column = 0; column < columns; column++)
                    {
                        if (board[row][column] < 0)
                        {
                            board[row][column] = 0;
                        }
                    }
                }

                return isComplete;
            }

            void Drop(int[][] board)
            {
                for (int column = 0; column < columns; column++)
                {
                    int lowestZeroRow = -1;

                    // Iterate over each column
                    for (int row = rows - 1; row >= 0; row--)
                    {
                        if (board[row][column] == 0)
                        {
                            lowestZeroRow = Math.Max(lowestZeroRow, row);
                        }
                        else if (lowestZeroRow >= 0)
                        {
                            int temp = board[row][column];
                            board[row][column] = board[lowestZeroRow][column];
                            board[lowestZeroRow][column] = temp;
                            lowestZeroRow--;
                        }
                    }
                }
            }


        }

















    }
}






























