// #include <gtest/gtest.h>
// #include <petscmat.h>

// #include <slepcbv.h>

// extern "C" {
// #include "../gcge/include/app_slepc.h"
// #include "../gcge/include/ops.h"
// }
// struct OPS_ dummy_ops = {};

// class MultiVecCreateTest : public ::testing::Test {
// protected:
//     static void TearDownTestSuite() {
//         PetscFinalize();
//     }
// };

// TEST_F(MultiVecCreateTest, CreateBVFromMatrix) {
//     BV bv = nullptr;
//     int num_vec = 3;

//     Mat testMat;
//     MatCreate(PETSC_COMM_WORLD, &testMat);
//     MatSetSizes(testMat, PETSC_DECIDE, PETSC_DECIDE, 5, 5);
//     MatSetUp(testMat);

//     // 测试函数
//     Testable_MultiVecCreateByMat(&bv, num_vec, testMat, &dummy_ops);

//     // 确保创建成功
//     ASSERT_NE(bv, nullptr);

//     PetscInt n, N, num;
//     BVGetSizes(bv, &n, &N, &num);

//     EXPECT_EQ(num, num_vec);
//     EXPECT_EQ(n, 5);

//     if (bv)
//         BVDestroy(&bv);
//     MatDestroy(&testMat);
// }

// int main(int argc, char **argv) {
// #if OPS_USE_SLEPC
//     SlepcInitialize(&argc, &argv, NULL, NULL);
// #elif OPS_USE_PETSC
//     PetscFunctionBeginUser;
//     PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
// #elif OPS_USE_MPI
//     MPI_Init(&argc, &argv);
// #endif
//     ::testing::InitGoogleTest(&argc, argv);
//     return RUN_ALL_TESTS();
// }
