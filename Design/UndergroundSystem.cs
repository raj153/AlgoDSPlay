using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.Design
{
    /*
    1396. Design Underground System
    https://leetcode.com/problems/design-underground-system/

    Approach 1: Two HashMaps

    Complexity Analysis
    •	Time complexity : O(1) for all.
        o	checkIn(...): Inserting a key/value pair into a HashMap is an O(1) operation.
        o	checkOut(...): Additionally, getting the corresponding value for a key from a HashMap is also an O(1) operation.
        o	getAverageTime(...): Dividing two numbers is also an O(1) operation.
    •	Space complexity : O(P+S2), where S is the number of stations on the network, and P is the number of passengers making a journey concurrently during peak time.
        o	The program uses two HashMaps. We need to determine the maximum sizes these could become.
        o	Firstly, we'll consider checkInData. This HashMap holds one entry for each passenger who has checkIn(...)ed, but not checkOut(...)ed. Therefore, the maximum size this HashMap could be is the maximum possible number of passengers making a journey at the same time, which we defined to be P. Therefore, the size of this HashMap is O(P).
        o	Secondly, we need to consider journeyData. This HashMap has one entry for each pair of stations that has had at least one passenger start and end a journey at those stations. Over time, we could reasonably expect every possible pair of the S stations on the network to have an entry in this HashMap, which would be O(S2).
        o	Seeing as we don't know whether S2 or P is larger, we need to add these together, giving a total space complexity of O(P+S2).
    */
    public class UndergroundSystem
    {
        private Dictionary<string, (double, double)> journeyData = new Dictionary<string, (double, double)>();
        private Dictionary<int, (string, int)> checkInData = new Dictionary<int, (string, int)>();
        public UndergroundSystem()

        {
        }

        public void CheckIn(int id, string stationName, int t)
        {
            checkInData[id] = (stationName, t);

        }
        public void CheckOut(int id, string stationName, int t)
        {
            // Look up the check in station and check in time for this id.
            // You could combine this "unpacking" into the other lines of code
            // to have less lines of code overall, but we've chosen to be verbose
            // here to make it easy for all learners to follow.
            (string startStation, int checkInTime) = checkInData[id];
            
            // Lookup the current travel time data for this route.
            string routeKey = StationsKey(startStation, stationName);
            (double totalTripTime, double totalTrips) = journeyData.ContainsKey(routeKey) ? journeyData[routeKey] : (0.0, 0.0);
            
            // Update the travel time data with this trip.
            double tripTime = t - checkInTime;
            journeyData[routeKey] = (totalTripTime + tripTime, totalTrips + 1);

            // Remove check in data for this id.
            // Note that this is optional, we'll talk about it in the space complexity analysis.
            checkInData.Remove(id);
        }

        public double GetAverageTime(string startStation, string endStation)
        {
            // Lookup how many times this journey has been made, and the total time.
            string routeKey = StationsKey(startStation, endStation);
            double totalTime = journeyData[routeKey].Item1;
            double totalTrips = journeyData[routeKey].Item2;
            // The average is simply the total divided by the number of trips.
            return totalTime / totalTrips;
        }

        private string StationsKey(string startStation, string endStation)
        {
            return startStation + "->" + endStation;
        }
    }
}