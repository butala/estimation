#pragma once

// ---------------------------------------------------------------------------
// Minimal dependency-free test harness.
//
//   TEST_CASE("name") { ... }        register a case
//   CHECK(cond, "what")              boolean check
//   CHECK_CLOSE(a, b, tol, "what")   floating point check (reports the error)
//   CHECK_MAT(A, B, tol, "what")     relative Frobenius check for matrices
//   CHECK_VEC(a, b, tol, "what")     relative 2-norm check for vectors
//   CHECK_THROWS([&]{...}, "what")   contract test: the body must throw
//
// CHECK_THROWS is how we prove the library's require() contracts actually
// fire. The library throws std::invalid_argument (never assert(), which is
// compiled out under NDEBUG) so the same check works in every build, and
// nanobind surfaces it to Python as ValueError.
// ---------------------------------------------------------------------------

#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include <exception>


namespace testing {

class Suite {
public:
    static Suite &get()
    {
        static Suite s;
        return s;
    }

    void add(std::string name, std::function<void()> body)
    {
        cases_.push_back({std::move(name), std::move(body)});
    }

    void expect(const bool ok, const char *what,
                const double err = 0.0, const double tol = 0.0)
    {
        if (ok) { ++passed_; return; }
        ++failed_;
        std::printf("    FAIL  %-56s err=%-11.3e tol=%.1e\n", what, err, tol);
    }

    void expect_throws(const std::function<void()> &body, const char *what)
    {
        bool threw = false;
        try {
            body();
        } catch (const std::exception &) {
            threw = true;
        }
        expect(threw, what, threw ? 0.0 : 1.0, 0.0);
    }

    int run()
    {
        for (auto &c : cases_) {
            std::printf("[ %s ]\n", c.name.c_str());
            std::fflush(stdout);
            c.body();
            std::fflush(stdout);
        }
        std::printf("\n%d checks passed, %d failed (%zu cases)\n",
                    passed_, failed_, cases_.size());
        std::printf("%s\n", failed_ ? "SOME CHECKS FAILED" : "ALL CHECKS PASSED");
        return failed_ ? 1 : 0;
    }

private:
    struct Case { std::string name; std::function<void()> body; };
    std::vector<Case> cases_;
    int passed_ = 0;
    int failed_ = 0;
};


struct Registrar {
    Registrar(const char *name, std::function<void()> body)
    {
        Suite::get().add(name, std::move(body));
    }
};


inline void
check(const bool ok, const char *what,
      const double err = 0.0, const double tol = 0.0)
{
    Suite::get().expect(ok, what, err, tol);
}

inline void
check_close(const double a, const double b, const double tol, const char *what)
{
    check(std::abs(a - b) <= tol, what, std::abs(a - b), tol);
}

inline void
check_throws(const std::function<void()> &body, const char *what)
{
    Suite::get().expect_throws(body, what);
}

} // namespace testing


#define TESTING_CAT2(a, b) a##b
#define TESTING_CAT(a, b) TESTING_CAT2(a, b)

#define TEST_CASE(name)                                                     \
    static void TESTING_CAT(testing_case_, __LINE__)();                     \
    static ::testing::Registrar TESTING_CAT(testing_reg_, __LINE__){        \
        name, TESTING_CAT(testing_case_, __LINE__)};                        \
    static void TESTING_CAT(testing_case_, __LINE__)()

#define CHECK(cond, what) ::testing::check(static_cast<bool>(cond), (what))
#define CHECK_CLOSE(a, b, tol, what) ::testing::check_close((a), (b), (tol), (what))
#define CHECK_THROWS(body, what) ::testing::check_throws((body), (what))
