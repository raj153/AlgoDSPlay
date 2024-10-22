using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay
{
    namespace Combinatoric.Enumeration
    {
        public static class CombinatoricEnumeration
        {
            // https://docs.python.org/2/library/itertools.html#itertools.combinations
            public static IEnumerable<List<T>> Combinations<T>(this IEnumerable<T> items, int r)
            {
                int n = items.Count();

                if (r > n) yield break;

                T[] pool = items.ToArray();
                int[] indices = Enumerable.Range(0, r).ToArray();

                yield return indices.Select(x => pool[x]).ToList();

                while (true)
                {
                    int i = indices.Length - 1;
                    while (i >= 0 && indices[i] == i + n - r)
                        i -= 1;

                    if (i < 0) yield break;

                    indices[i] += 1;

                    for (int j = i + 1; j < r; j += 1)
                        indices[j] = indices[j - 1] + 1;

                    yield return indices.Select(x => pool[x]).ToList();
                }
            }

            public static IEnumerable<List<T>> Permutations<T>(this IEnumerable<T> items) => items.Permutations(items.Count());

            // https://docs.python.org/2/library/itertools.html#itertools.permutations
            public static IEnumerable<List<T>> Permutations<T>(this IEnumerable<T> items, int r)
            {
                int n = items.Count();

                if (r > n) yield break;

                T[] pool = items.ToArray();
                int[] indices = Enumerable.Range(0, n).ToArray();
                int[] cycles = Enumerable.Range(n - r + 1, r).Reverse().ToArray();

                yield return indices.Take(r).Select(x => pool[x]).ToList();

                while (true)
                {
                    int i = cycles.Length - 1;
                    while (i >= 0)
                    {
                        cycles[i] -= 1;
                        if (cycles[i] == 0)
                        {
                            // rotate indices from i to end
                            int tempInt = indices[i];
                            int[] tmpArray = indices.Skip(i).ToArray();
                            tmpArray.CopyTo(indices, i);
                            indices[indices.Length - 1] = tempInt;
                            cycles[i] = n - i;
                        }
                        else
                        {
                            int j = indices.Length - cycles[i];
                            (indices[i], indices[j]) = (indices[j], indices[i]);
                            yield return indices.Take(r).Select(x => pool[x]).ToList();
                            break;
                        }
                        i -= 1;
                    }
                    if (i < 0) yield break;
                }
            }
        }

        // based on http://stackoverflow.com/a/3486820/1858296
        public static class SortedDictionaryExtensions
        {
            private static Tuple<int, int> GetPossibleIndices<TKey, TValue>(SortedDictionary<TKey, TValue> dictionary, TKey key, bool strictlyDifferent, out List<TKey> list)
            {
                list = dictionary.Keys.ToList();
                int index = list.BinarySearch(key, dictionary.Comparer);
                if (index >= 0)
                {
                    // exists
                    if (strictlyDifferent)
                        return Tuple.Create(index - 1, index + 1);
                    else
                        return Tuple.Create(index, index);
                }
                else
                {
                    // doesn't exist
                    int indexOfBiggerNeighbour = ~index; //bitwise complement of the return value

                    if (indexOfBiggerNeighbour == list.Count)
                    {
                        // bigger than all elements
                        return Tuple.Create(list.Count - 1, list.Count);
                    }
                    else if (indexOfBiggerNeighbour == 0)
                    {
                        // smaller than all elements
                        return Tuple.Create(-1, 0);
                    }
                    else
                    {
                        // Between 2 elements
                        int indexOfSmallerNeighbour = indexOfBiggerNeighbour - 1;
                        return Tuple.Create(indexOfSmallerNeighbour, indexOfBiggerNeighbour);
                    }
                }
            }

            public static TKey LowerKey<TKey, TValue>(this SortedDictionary<TKey, TValue> dictionary, TKey key)
            {
                List<TKey> list;
                var indices = GetPossibleIndices(dictionary, key, true, out list);
                if (indices.Item1 < 0)
                    return default(TKey);

                return list[indices.Item1];
            }
            public static KeyValuePair<TKey, TValue> LowerEntry<TKey, TValue>(this SortedDictionary<TKey, TValue> dictionary, TKey key)
            {
                List<TKey> list;
                var indices = GetPossibleIndices(dictionary, key, true, out list);
                if (indices.Item1 < 0)
                    return default(KeyValuePair<TKey, TValue>);

                var newKey = list[indices.Item1];
                return new KeyValuePair<TKey, TValue>(newKey, dictionary[newKey]);
            }

            public static TKey FloorKey<TKey, TValue>(this SortedDictionary<TKey, TValue> dictionary, TKey key)
            {
                List<TKey> list;
                var indices = GetPossibleIndices(dictionary, key, false, out list);
                if (indices.Item1 < 0)
                    return default(TKey);

                return list[indices.Item1];
            }
            public static KeyValuePair<TKey, TValue> FloorEntry<TKey, TValue>(this SortedDictionary<TKey, TValue> dictionary, TKey key)
            {
                List<TKey> list;
                var indices = GetPossibleIndices(dictionary, key, false, out list);
                if (indices.Item1 < 0)
                    return default(KeyValuePair<TKey, TValue>);

                var newKey = list[indices.Item1];
                return new KeyValuePair<TKey, TValue>(newKey, dictionary[newKey]);
            }

            public static TKey CeilingKey<TKey, TValue>(this SortedDictionary<TKey, TValue> dictionary, TKey key)
            {
                List<TKey> list;
                var indices = GetPossibleIndices(dictionary, key, false, out list);
                if (indices.Item2 == list.Count)
                    return default(TKey);

                return list[indices.Item2];
            }
            public static KeyValuePair<TKey, TValue> CeilingEntry<TKey, TValue>(this SortedDictionary<TKey, TValue> dictionary, TKey key)
            {
                List<TKey> list;
                var indices = GetPossibleIndices(dictionary, key, false, out list);
                if (indices.Item2 == list.Count)
                    return default(KeyValuePair<TKey, TValue>);

                var newKey = list[indices.Item2];
                return new KeyValuePair<TKey, TValue>(newKey, dictionary[newKey]);
            }

            public static TKey HigherKey<TKey, TValue>(this SortedDictionary<TKey, TValue> dictionary, TKey key)
            {
                List<TKey> list;
                var indices = GetPossibleIndices(dictionary, key, true, out list);
                if (indices.Item2 == list.Count)
                    return default(TKey);

                return list[indices.Item2];
            }
            public static KeyValuePair<TKey, TValue> HigherEntry<TKey, TValue>(this SortedDictionary<TKey, TValue> dictionary, TKey key)
            {
                List<TKey> list;
                var indices = GetPossibleIndices(dictionary, key, true, out list);
                if (indices.Item2 == list.Count)
                    return default(KeyValuePair<TKey, TValue>);

                var newKey = list[indices.Item2];
                return new KeyValuePair<TKey, TValue>(newKey, dictionary[newKey]);
            }
        }

    }

}