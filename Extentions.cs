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
    }

}