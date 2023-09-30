using System.Text;
using System.Runtime.InteropServices;
using System.Globalization;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

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
                reverseArrayInPlance(redShirtSpeeds);
            
            int totalSpeed =0; 
            for(int idx=0; idx< redShirtSpeeds.Length; idx++){
                int rider1 = redShirtSpeeds[idx];
                int rider2 = blueShirtSpeeds[idx];
                totalSpeed += Math.Max(rider1, rider2);
            }
            return totalSpeed;

        }

        private void reverseArrayInPlance(int[] array)
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
    }
}