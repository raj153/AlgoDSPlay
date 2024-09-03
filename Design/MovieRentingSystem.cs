using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.Design
{
    /*
    1912. Design Movie Rental System
    https://leetcode.com/problems/design-movie-rental-system/

    Complexity:
    Time:
        Constructor: O(NlogN)
        search, rent, drop, report: O(logN)
    Space: O(N)

    */
    public class MovieRentingSystem
    {
        Dictionary<int, SortedSet<Entry>> unrented = new();
        SortedSet<Entry> rented = new(new CustomCompare());
        Dictionary<Tuple<int, int>, int> prices = new();

        public MovieRentingSystem(int n, int[][] entries)
        {
            foreach (int[] entry in entries)
            {
                int shop = entry[0];
                int movie = entry[1];
                int price = entry[2];

                if (!unrented.ContainsKey(movie))
                    unrented.Add(movie, new SortedSet<Entry>(new CustomCompare()));
                unrented[movie].Add(new Entry(price, shop, movie));
                prices.Add(Tuple.Create(shop, movie), price);
            }
        }

        public IList<int> Search(int movie)
        {
            if (!unrented.ContainsKey(movie))
                return new List<int>();
            IList<int> result = new List<int>();
            int i = 0;
            foreach (Entry e in unrented[movie])
            {
                result.Add(e.shop);
                ++i;
                if (i == 5) break;
            }
            return result;
        }

        public void Rent(int shop, int movie)
        {
            int price = prices[Tuple.Create(shop, movie)];
            unrented[movie].Remove(new Entry(price, shop, movie));
            rented.Add(new Entry(price, shop, movie));
        }

        public void Drop(int shop, int movie)
        {
            int price = prices[Tuple.Create(shop, movie)];
            unrented[movie].Add(new Entry(price, shop, movie));
            rented.Remove(new Entry(price, shop, movie));
        }

        public IList<IList<int>> Report()
        {
            IList<IList<int>> result = new List<IList<int>>();
            int i = 0;
            foreach (Entry e in rented)
            {
                result.Add(new List<int> { e.shop, e.movie });
                ++i;
                if (i == 5) break;
            }
            return result;
        }
    }

    public class Entry
    {
        public int price, shop, movie;
        public Entry(int price, int shop, int movie)
        {
            this.price = price;
            this.shop = shop;
            this.movie = movie;
        }
    }

    public class CustomCompare : IComparer<Entry>
    {
        public int Compare(Entry a, Entry b)
        {
            if (a.price != b.price)
                return a.price - b.price;
            if (a.shop != b.shop)
                return a.shop - b.shop;
            return a.movie - b.movie;
        }
    }

    /**
 * Your MovieRentingSystem object will be instantiated and called as such:
 * MovieRentingSystem obj = new MovieRentingSystem(n, entries);
 * IList<int> param_1 = obj.Search(movie);
 * obj.Rent(shop,movie);
 * obj.Drop(shop,movie);
 * IList<IList<int>> param_4 = obj.Report();
 */
}