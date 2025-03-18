  This directory contains boost/mp11 headers adapted for use in SYCL headers in
  a way that does not conflict with potential use of boost in user code.
  Particularly, `BOOST_*` macros are replaced with `SYCL_DETAIL_BOOST_*`, APIs
  are moved into the top-level `sycl::detail` namespace. For example,
  `sycl::detail::boost::mp11::mp_list`.
