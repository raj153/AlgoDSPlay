using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.DataStructures
{
    public class SegmentTree
    {
        private int[] tree;
        private int n;
        //init tree
        public SegmentTree(int n){
            this.n=n;
            int height=(int)(Math.Ceiling(Math.Log(n)/Math.Log(2)));
            int size=2*((int)Math.Pow(2,height)) -1;
            tree=new int[size];
        }
        
        public int query(int left,int right){
            return query(0,0,n-1,left,right);
        }
        
        private int query(int idx,int l,int r,int left,int right){
            if(r<left||right<l){
                return 0;
            }else if(left<=l && r<=right){
                return tree[idx];
            }else{
                int m=l+(r-l)/2;
                int lMax=query(2*idx+1,l,m,left,right);
                int rMax=query(2*idx+2,m+1,r,left,right);
                return Math.Max(lMax,rMax);
            }            
        }
        
        public void update(int idx,int val){
            update(0,0,n-1,idx,val);
        }
        
        private void update(int idx,int l,int r,int updateIdx,int val){
            if(l==r){
                tree[idx]=Math.Max(tree[idx],val);    
                return;
            }
            
            int m=l+(r-l)/2;
            if(l<=updateIdx && updateIdx<=m){
                update(2*idx+1,l,m,updateIdx,val);
            }else {
                update(2*idx+2,m+1,r,updateIdx,val);
            }
            tree[idx]=Math.Max(tree[2*idx+1],tree[2*idx+2]);
        }
    
        public int max(){
            return tree[0];
        }

    }
}