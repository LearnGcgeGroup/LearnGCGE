#include <gtest/gtest.h>

// 被测试的函数
int add(int a, int b) {
    return a + b;
}

// 测试用例
TEST(AdditionTest, HandlesPositiveNumbers) { // TEST(test_case_name, test_name)。GoogleTest 允许在一个测试用例组 (AdditionTest) 里定义多个测试，每个测试都有不同的 test_name
    EXPECT_EQ(add(2, 3), 5);                 // EXPECT_EQ(actual, expected) 用于断言实际值 (actual) 是否等于期望值 (expected)。
}

TEST(AdditionTest, HandlesNegativeNumbers) {
    EXPECT_EQ(add(-2, -3), -5);
}

TEST(AdditionTest, SimpleTest) {
    ASSERT_TRUE(1 > 2);
}

// 主函数
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
