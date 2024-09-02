using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.Design
{
    /*
    2254. Design Video Sharing Platform
    https://leetcode.com/problems/design-video-sharing-platform/
    Complexity
    •	Time complexity: O(1) for insertion, O(logN) for retrieval, where N is the count of already removed videoIds, not yet re-used (see Approach section)
    •	Space complexity: O(n+N) where n is the number of videos stored and N is the count of already removed videoIds, not yet re-used (see Approach section)

    */
    public class Video
    {
        public int Id { get; set; }
        public string Content { get; set; }
        public int Likes { get; set; }
        public int Dislikes { get; set; }
        public int WatchCount { get; set; }

        public Video(int id, string content)
        {
            Id = id;
            Content = content;
        }
    }

    public class VideoSharingPlatform
    {
        private int _lastNewVideoId;
        private Dictionary<int, Video> _videos;
        private PriorityQueue<int, int> _removedVideos;

        public VideoSharingPlatform()
        {
            _lastNewVideoId = 0;
            _videos = new Dictionary<int, Video>();
            _removedVideos = new PriorityQueue<int, int>();
        }

        public int Upload(string video)
        {
            var nextVideoId = _lastNewVideoId;
            if (_removedVideos.Count > 0)
            {
                nextVideoId = _removedVideos.Dequeue();
            }

            if (_videos.TryAdd(nextVideoId, new Video(nextVideoId, video)))
                return nextVideoId == _lastNewVideoId ? _lastNewVideoId++ : nextVideoId;
            return -1;
        }

        public void Remove(int videoId)
        {
            if (!HasVideo(videoId)) return;

            _videos.Remove(videoId);
            _removedVideos.Enqueue(videoId, videoId);
        }

        public string Watch(int videoId, int startMinute, int endMinute)
        {
            if (!HasVideo(videoId)) return "-1";

            var vidLen = _videos[videoId].Content.Length;
            _videos[videoId].WatchCount++;
            return _videos[videoId]
                .Content
                .Substring(
                    startMinute,
                    Math.Min(vidLen - startMinute, endMinute - startMinute + 1)
                    );
        }

        public void Like(int videoId)
        {
            if (!HasVideo(videoId)) return;

            _videos[videoId].Likes++;
        }

        public void Dislike(int videoId)
        {
            if (!HasVideo(videoId)) return;

            _videos[videoId].Dislikes++;
        }

        public int[] GetLikesAndDislikes(int videoId)
        {
            if (!HasVideo(videoId)) return new int[] { -1 };

            var video = _videos[videoId];

            return new int[] { video.Likes, video.Dislikes };
        }

        public int GetViews(int videoId)
        {
            if (!HasVideo(videoId)) return -1;

            return _videos[videoId].WatchCount;
        }

        private bool HasVideo(int videoId)
        {
            return _videos.ContainsKey(videoId);
        }
    }

    /**
     * Your VideoSharingPlatform object will be instantiated and called as such:
     * VideoSharingPlatform obj = new VideoSharingPlatform();
     * int param_1 = obj.Upload(video);
     * obj.Remove(videoId);
     * string param_3 = obj.Watch(videoId,startMinute,endMinute);
     * obj.Like(videoId);
     * obj.Dislike(videoId);
     * int[] param_6 = obj.GetLikesAndDislikes(videoId);
     * int param_7 = obj.GetViews(videoId);
     */

}