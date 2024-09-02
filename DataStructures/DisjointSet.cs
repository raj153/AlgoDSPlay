using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.DataStructures
{
    //UnionFind/DisjointSet - Another variation
    //https://www.geeksforgeeks.org/introduction-to-disjoint-set-data-structure-or-union-find-algorithm/
    public class DisjointSet
    {
        public List<int> parent = new();
        List<int> size = new();
        public DisjointSet(int V)
        {
            for (int i = 0; i < V; i++)
            {
                parent.Add(i);
                size.Add(1);
            }
        }

        public int FindUParent(int node)
        {
            if (node == parent[node])
                return node;
            return parent[node] = FindUParent(parent[node]);
        }

        public void UnionBySize(int u, int v)
        {
            int ulpu = FindUParent(u);
            int ulpv = FindUParent(v);
            if (ulpu == ulpv)
                return;

            if (size[ulpu] < size[ulpv])
            {
                parent[ulpu] = ulpv;
                size[ulpv] = size[ulpv] + size[ulpu];
            }
            else
            {
                parent[ulpv] = ulpu;
                size[ulpu] = size[ulpu] + size[ulpv];
            }
        }
    }
    //DisjointSet using Arrays
    public class DSUArray
    {
        int[] parent; 
        int[] rank;

        public DSUArray(int size)
        {
            parent = new int[size];
            rank = new int[size];
            
            for (int i = 0; i < size; i++) {
                parent[i] = i;
                rank[i] =0;
            }

            
        }

        public int Find(int x)
        {
            if (parent[x] != x) parent[x] = Find(parent[x]);
            return parent[x];
        }

        public bool Union(int x, int y)
        {
            int xr = Find(x), yr = Find(y);
            if (xr == yr)
            {
                return false;
            }
            else if (rank[xr] < rank[yr])
            {
                parent[xr] = yr;
            }
            else if (rank[xr] > rank[yr])
            {
                parent[yr] = xr;
            }
            else
            {
                parent[yr] = xr;
                rank[xr]++;
            }
            return true;
        }

        internal int find(object value)
        {
            throw new NotImplementedException();
        }
    }

}
