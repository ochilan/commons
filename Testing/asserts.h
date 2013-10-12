#ifndef ASSERTS_H_
#define ASSERTS_H_

#ifdef _MSC_VER
# define BOOST_ASSERT_MSG_OSTREAM std::cout
#else
# define BOOST_ASSERT_MSG_OSTREAM std::cerr
#endif
#include <boost/assert.hpp>

#define asserts(expr, msg) (BOOST_ASSERT_MSG((expr), (msg)))
//#define asserts(expr, msg) ((void)0)

#endif /* ASSERTS_H_ */
