import pytest

from falkon.utils.helpers import select_dim_over_nm_v2, select_dim_over_nm, select_dim_over_nd


def do_check(created, available, d0, max_d0, d1, max_d1):
    assert created <= available, "Too much memory allocated"
    assert d0 <= max_d0, "d0 greater than allowed"
    assert d1 <= max_d1, "d1 greater than allowed"


class TestDimOverNM:
    @pytest.mark.parametrize("avail_mem", [0.1 * 2**20, 10 * 2**40])
    def test_dim_over_nm_v2(self, avail_mem):
        tot_n = 40_000
        tot_m = 2_000
        tot_d = 30_720
        tot_t = 10

        n, m = select_dim_over_nm_v2(
            tot_n, tot_m, coef_nm=1.0, coef_n=tot_d + tot_t, coef_m=tot_d + tot_t, rest=0, max_mem=avail_mem
        )
        created = n * m + n * tot_t + n * tot_d + m * tot_d + m * tot_t
        do_check(created, avail_mem, n, tot_n, m, tot_m)

    @pytest.mark.parametrize("avail_mem", [0.1 * 2**20, 10 * 2**40])
    def test_dim_over_nm(self, avail_mem):
        tot_n = 40_000
        tot_m = 2_000
        tot_d = 30_720
        n, m = select_dim_over_nm(
            max_n=tot_n,
            max_m=tot_m,
            d=tot_d,
            coef_nd=0.1,
            coef_md=0.2,
            coef_nm=0.3,
            coef_n=0.4,
            coef_m=5,
            rest=6000,
            max_mem=avail_mem,
        )
        created = n * tot_d * 0.1 + m * tot_d * 0.2 + n * m * 0.3 + n * 0.4 + m * 5 + 6000
        do_check(created, avail_mem, n, tot_n, m, tot_m)

    @pytest.mark.parametrize("avail_mem", [0.5 * 2**30, 10 * 2**40])
    def test_dim_over_nm_v2_zero(self, avail_mem):
        tot_n = 400_000
        tot_m = 2_000
        n, m = select_dim_over_nm_v2(tot_n, tot_m, coef_nm=0, coef_n=0, coef_m=0, rest=0, max_mem=avail_mem)
        created = 0
        do_check(created, avail_mem, n, tot_n, m, tot_m)

        n, m = select_dim_over_nm_v2(tot_n, tot_m, coef_nm=1.3, coef_n=0, coef_m=0, rest=9890, max_mem=avail_mem)
        created = 1.3 * n * m + 9890
        do_check(created, avail_mem, n, tot_n, m, tot_m)

        n, m = select_dim_over_nm_v2(tot_n, tot_m, coef_nm=0, coef_n=2.0, coef_m=0, rest=9890, max_mem=avail_mem)
        created = 2 * n + 9890
        do_check(created, avail_mem, n, tot_n, m, tot_m)

    @pytest.mark.parametrize("avail_mem", [0.5 * 2**30, 10 * 2**40])
    def test_dim_over_nm_zero(self, avail_mem):
        tot_n = 40_000
        tot_m = 2_000
        tot_d = 30_720
        n, m = select_dim_over_nm(
            max_n=tot_n,
            max_m=tot_m,
            d=tot_d,
            coef_nd=0,
            coef_md=0,
            coef_nm=0,
            coef_n=0,
            coef_m=0,
            rest=0,
            max_mem=avail_mem,
        )
        created = 0
        do_check(created, avail_mem, n, tot_n, m, tot_m)

        n, m = select_dim_over_nm(
            max_n=tot_n,
            max_m=tot_m,
            d=tot_d,
            coef_nd=10,
            coef_md=0,
            coef_nm=0,
            coef_n=0,
            coef_m=0,
            rest=0,
            max_mem=avail_mem,
        )
        created = 10 * n * tot_d
        do_check(created, avail_mem, n, tot_n, m, tot_m)

        n, m = select_dim_over_nm(
            max_n=tot_n,
            max_m=tot_m,
            d=tot_d,
            coef_nd=0,
            coef_md=0,
            coef_nm=5,
            coef_n=0,
            coef_m=0,
            rest=9890,
            max_mem=avail_mem,
        )
        created = 5 * n * m + 9890
        do_check(created, avail_mem, n, tot_n, m, tot_m)

    @pytest.mark.parametrize("avail_mem", [32])
    def test_dim_over_nm_v2_notenough(self, avail_mem):
        tot_n = 40_000
        tot_m = 2_000
        tot_d = 30_720
        tot_t = 10
        with pytest.raises(MemoryError):
            select_dim_over_nm_v2(
                tot_n, tot_m, coef_nm=1.0, coef_n=tot_d + tot_t, coef_m=tot_d + tot_t, rest=0, max_mem=avail_mem
            )
        with pytest.raises(MemoryError):
            select_dim_over_nm_v2(tot_n, tot_m, coef_nm=0.1, coef_n=0, coef_m=0, rest=12312, max_mem=avail_mem)

    @pytest.mark.parametrize("avail_mem", [32])
    def test_dim_over_nm_notenough(self, avail_mem):
        tot_n = 40_000
        tot_m = 2_000
        tot_d = 30_720
        tot_t = 10
        with pytest.raises(MemoryError):
            select_dim_over_nm(
                max_n=tot_n,
                max_m=tot_m,
                d=tot_d,
                coef_nd=1.3,
                coef_md=tot_d,
                coef_nm=2 + tot_t,
                coef_n=0,
                coef_m=1,
                rest=0,
                max_mem=avail_mem,
            )
        with pytest.raises(MemoryError):
            select_dim_over_nm(
                max_n=tot_n,
                max_m=tot_m,
                d=tot_d,
                coef_nd=0,
                coef_md=0,
                coef_nm=1,
                coef_n=4,
                coef_m=0,
                rest=1,
                max_mem=avail_mem + 1,
            )
        with pytest.raises(MemoryError):
            select_dim_over_nm(
                max_n=tot_n,
                max_m=tot_m,
                d=tot_d,
                coef_nd=0,
                coef_md=0,
                coef_nm=avail_mem,
                coef_n=0,
                coef_m=0,
                rest=1,
                max_mem=avail_mem,
            )


class TestDimOverND:
    @pytest.mark.parametrize("avail_mem", [1 * 2**20, 10 * 2**40])
    def test_dim_over_nd(self, avail_mem):
        tot_d = 1231
        tot_n = 3700000
        n, d = select_dim_over_nd(
            max_n=tot_n, max_d=tot_d, coef_nd=1.2, coef_n=1.3, coef_d=1.4, rest=98765, max_mem=avail_mem
        )
        created = n * d * 1.2 + n * 1.3 + d * 1.4 + 98765
        do_check(created, avail_mem, n, tot_n, d, tot_d)

        n, d = select_dim_over_nd(
            max_n=tot_n, max_d=tot_d, coef_nd=20, coef_n=1.3, coef_d=1.4, rest=987650, max_mem=avail_mem
        )
        created = n * d * 20 + n * 1.3 + d * 1.4 + 987650
        do_check(created, avail_mem, n, tot_n, d, tot_d)

    @pytest.mark.parametrize("avail_mem", [1 * 2**20, 10 * 2**40])
    def test_dim_over_nd_zero(self, avail_mem):
        tot_d = 1231
        tot_n = 3700000
        n, d = select_dim_over_nd(max_n=tot_n, max_d=tot_d, coef_nd=0, coef_n=0, coef_d=0, rest=0, max_mem=avail_mem)
        created = 0
        do_check(created, avail_mem, n, tot_n, d, tot_d)

        n, d = select_dim_over_nd(max_n=tot_n, max_d=tot_d, coef_nd=0, coef_n=34, coef_d=20, rest=0, max_mem=avail_mem)
        created = n * 34 + d * 20
        do_check(created, avail_mem, n, tot_n, d, tot_d)

        n, d = select_dim_over_nd(max_n=tot_n, max_d=tot_d, coef_nd=0, coef_n=324, coef_d=0, rest=0, max_mem=avail_mem)
        created = n * 324
        do_check(created, avail_mem, n, tot_n, d, tot_d)

    @pytest.mark.parametrize("avail_mem", [32])
    def test_dim_over_nd_notenough(self, avail_mem):
        tot_d = 1231
        tot_n = 3700000
        with pytest.raises(MemoryError):
            select_dim_over_nd(
                max_n=tot_n, max_d=tot_d, coef_nd=1.2, coef_n=1.3, coef_d=1.4, rest=98765, max_mem=avail_mem
            )
        with pytest.raises(MemoryError):
            select_dim_over_nd(
                max_n=tot_n, max_d=tot_d, coef_nd=avail_mem, coef_n=0, coef_d=0, rest=1, max_mem=avail_mem
            )
        with pytest.raises(MemoryError):
            select_dim_over_nd(
                max_n=tot_n, max_d=tot_d, coef_nd=0, coef_n=0.01, coef_d=0, rest=98765, max_mem=avail_mem
            )
