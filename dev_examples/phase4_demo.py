def test_normalized_content_deduplication(self):
        """Test content normalization for deduplication."""
        posts = [
            create_test_post(1, "Hello World! #test @user1"),
            create_test_post(2, "hello world test"),  # Normalized same as 1 without user1
            create_test_post(3, "Hello, World! Test @user1!"),  # Also normalized same as 1
            create_test_post(4, "Different content here")
        ]
        
        dedup = Deduplicator()
        unique_posts, stats = dedup.deduplicate(posts)
        
        assert len(unique_posts) == 2  # Only 2 unique after normalization
        assert stats['duplicates_removed'] == 2