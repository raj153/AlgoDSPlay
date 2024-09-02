using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.DataStructures
{
    public class MinHeapForAStarAlgo
    {
        List<NodeExt> heap = new List<NodeExt>();
        Dictionary<string, int> nodePositionInHeap = new Dictionary<string, int>();

        public MinHeapForAStarAlgo(List<NodeExt> array){
            for(int i=0; i< array.Count; i++){
                NodeExt node = array[i];
                nodePositionInHeap[node.Id]=i;
            }
            heap = BuildHeap(array);
        }

        //T:O(n) | S:O(1)
        private List<NodeExt> BuildHeap(List<NodeExt> array)
        {
            int firstParentIdx = (array.Count-2)/2;
            for(int currentIdx=firstParentIdx+1; currentIdx >=0; currentIdx--){
                SiftDown(currentIdx, array.Count-1, array);
            }
            return array;
        }
        public bool IsEmpty(){
            return heap.Count == 0;
        }

        //T:O(log(n)) | S:O(1)
        private void SiftDown(int currentIdx, int endIdx, List<NodeExt> array)
        {
            int childOneIdx = currentIdx*2+1;
            while(childOneIdx <= endIdx){
                int childTwoIdx = currentIdx *2+2 <= endIdx? currentIdx*2+2:-1;
                int idxToSwap;
                if(childTwoIdx != -1 && 
                    array[childTwoIdx].estimatedDistanceToEnd < array[childOneIdx].estimatedDistanceToEnd){
                        idxToSwap = childTwoIdx;
                    }else {idxToSwap = childOneIdx;}
                if(array[idxToSwap].estimatedDistanceToEnd < array[currentIdx].estimatedDistanceToEnd){
                    Swap(currentIdx,idxToSwap);
                    currentIdx = idxToSwap;
                    childOneIdx = currentIdx*2+1;

                }
                else{
                    return;
                }

            }

        }
        //T:O(log(n) | S:O(1)
        private void SiftUp(int currentIdx){
            int parentIdx= (currentIdx-1)/2;
            while(currentIdx >0 && heap[currentIdx].estimatedDistanceToEnd < heap[parentIdx].estimatedDistanceToEnd){
                Swap(currentIdx,parentIdx);
                currentIdx = parentIdx;
                parentIdx = (currentIdx-1)/2;
            }
        }
        
        public NodeExt Remove(){
            if(IsEmpty()) return null;

            Swap(0, heap.Count-1);
            NodeExt node = heap[heap.Count-1];
            heap.RemoveAt(heap.Count-1);
            nodePositionInHeap.Remove(node.Id);
            SiftDown(0, heap.Count-1, heap);
            return node;

        }
        public void Insert(NodeExt node){
            heap.Add(node);
            nodePositionInHeap[node.Id]=heap.Count-1;
            SiftUp(heap.Count-1);
        }
        public void Update(NodeExt node){
            SiftUp(nodePositionInHeap[node.Id]);
        }
        public bool ContainsNode(NodeExt node){
            return nodePositionInHeap.ContainsKey(node.Id);
        }
        private void Swap(int idx1, int idx2)
        {
            nodePositionInHeap[heap[idx1].Id]= idx2;
            nodePositionInHeap[heap[idx2].Id]= idx1;

            NodeExt temp = heap[idx1];
            heap[idx1] = heap[idx2];
            heap[idx2] = temp;

            
        }
    }
}