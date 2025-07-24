CREATE DATABASE VidAffinityDB;

CREATE TABLE user_video_affinity (
    user_id VARCHAR(36) NOT NULL,
    video_id VARCHAR(24) NOT NULL,
    affinity_score FLOAT CHECK (affinity_score BETWEEN 0 AND 1),
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, video_id)
);