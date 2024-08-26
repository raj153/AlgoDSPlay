using System.Xml;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.Graph
{
    public class Cycles
    {
        //https://www.algoexpert.io/questions/cycle-in-graph
        public static bool CycleInGrah(int[][] edges){
            //T:O(v+e) | S: O(v)
            bool result = CycleInGrah1(edges);

            //T:O(v+e) | S: O(v)
            result = CycleInGrah2(edges);
            return false;
 
        }

        static int WHITE=0; //unvisited, 
        static int GREY=1; //Visited and Instack 
        static int  BLACK=2; //Visited and Finished traversing 
        private static bool CycleInGrah2(int[][] edges)
        {
            int numberOfNodes = edges.Length;
            int[] colors = new int[numberOfNodes];
            Array.Fill(colors, WHITE);
            
            for(int node=0; node < edges.Length; node++){
                if(colors[node] != WHITE) continue;

                bool containsCycle = TraverseAndColorNodes(node, edges, colors);
                if(containsCycle) return true;
            }
            return false;
        }

        private static bool TraverseAndColorNodes(int node, int[][] edges, int[] colors)
        {
            colors[node]=GREY;
            int[] neighbors = edges[node];
            foreach(var neighbor in neighbors){
                int neighborColor = colors[neighbor];

                if (neighborColor == GREY) // Visited and an ancestor 
                    return true;
                if(neighborColor == BLACK)
                    continue;
                
                bool containsCycle= TraverseAndColorNodes(neighbor, edges, colors);
                if(containsCycle) return true;
            }
            colors[node] = BLACK;
            return false;
        }

        private static bool CycleInGrah1(int[][] edges){
            int numberOfNodes = edges.Length;
            bool[] visited = new bool[numberOfNodes];
            bool[] currentlyInStack = new bool[numberOfNodes];
            Array.Fill(visited, false);
            Array.Fill(currentlyInStack, false);

            for(int node=0; node < numberOfNodes; node++){
                if(visited[node])
                    continue;
                
                bool containsCycle = IsNodeInCycle(node, edges,visited,currentlyInStack);
                if(containsCycle) return true;
            }
            return false;       
        }

        private static bool IsNodeInCycle(int node, int[][] edges, bool[] visited, bool[] currentlyInStack)
        {
            visited[node] = true;
            currentlyInStack[node]=true;

            bool containsCycle = false;
            int[] neighbors = edges[node];
            foreach(var neighbor in neighbors){
                if(!visited[neighbor]){
                    containsCycle = IsNodeInCycle(neighbor, edges, visited, currentlyInStack);
                
                    if(containsCycle)
                        return true;
                }
                else if(currentlyInStack[neighbor]) return true;
            }
            currentlyInStack[node]= false;
            return containsCycle;


        }
    }
}