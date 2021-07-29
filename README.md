# Detect Similarity in Yoga Pose using Alphapose Model

[Alphapose Pose Estimaion Model](https://github.com/MVIG-SJTU/AlphaPose) AlphaPose is an accurate multi-person pose estimator, which is the first open-source system that achieves 70+ mAP (75 mAP) on COCO dataset and 80+ mAP (82.1 mAP) on MPII dataset. To match poses that correspond to the same person across frames, we also provide an efficient online pose tracker called Pose Flow. It is the first open-source online pose tracker that achieves both 60+ mAP (66.5 mAP) and 50+ MOTA (58.3 MOTA) on PoseTrack Challenge dataset.

### Yoga Pose Simlirity Detection

This Flask application mainly helps in Detetcting Similarity between user input Yoga Pose and Ideal Assan Pose and Returns a Similarity score for those two Poses.
Application in NutShell :-
1) User selects a Assan(Pose) Name form drop down List.
2) Uploads the image of their Assan Pose.
3) Application then calls Alphapose Model to get pose keyjoints for a given user image and compares it with exsisting ideal assan pose keyjoints. The comparision of two pose is done using Cosine Similarity and finally app returns a similairty score for two pose with their pose keyjoints projection on User Input Image and Ideal Assan Image to visually judge a difference between the two pose. 

![Yoga Pose Simlirity App Demo](https://github.com/PalashShinde/Detect_Yoga_Pose_With_AI/blob/main/app/gifs/yoga_cut_version.gif)

### Try it now
