<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="X-UA-Compatible" content="IE=edge">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="">
<meta name="author" content="">
<link rel="icon" href="../../favicon.ico">

<title>AtCoder Problems</title>

<!-- Bootstrap core CSS -->
<link href="css/bootstrap.min.css" rel="stylesheet">
<!-- Bootstrap theme -->
<link href="css/bootstrap-theme.min.css" rel="stylesheet">
<link href="css/dataTables.bootstrap.css" rel="stylesheet">

<!-- Custom styles for this template -->
<link href="theme.css" rel="stylesheet">


<!-- HTML5 shim and Respond.js for IE8 support of HTML5 elements and media queries -->
<!--[if lt IE 9]>
      <script src="https://oss.maxcdn.com/html5shiv/3.7.2/html5shiv.min.js"></script>
      <script src="https://oss.maxcdn.com/respond/1.4.2/respond.min.js"></script>
    <![endif]-->

<script>
  (function(i,s,o,g,r,a,m){i['GoogleAnalyticsObject']=r;i[r]=i[r]||function(){
  (i[r].q=i[r].q||[]).push(arguments)},i[r].l=1*new Date();a=s.createElement(o),
  m=s.getElementsByTagName(o)[0];a.async=1;a.src=g;m.parentNode.insertBefore(a,m)
  })(window,document,'script','//www.google-analytics.com/analytics.js','ga');

  ga('create', 'UA-60720107-1', 'auto');
  ga('send', 'pageview');

</script>
</head>

<body role="document">
	<!-- Fixed navbar -->
	<nav class="navbar navbar-inverse navbar-fixed-top">
		<div class="container">
			<div class="navbar-header">
				<button type="button" class="navbar-toggle collapsed"
					data-toggle="collapse" data-target="#navbar" aria-expanded="false"
					aria-controls="navbar">
					<span class="sr-only">Toggle navigation</span> <span
						class="icon-bar"></span> <span class="icon-bar"></span> <span
						class="icon-bar"></span>
				</button>
				<span class="navbar-brand">AtCoder Problems</span>
			</div>
			<div id="navbar" class="navbar-collapse collapse">
				<ul class="nav navbar-nav">
					<li
						<?php if (!isset ($_GET ["ranking"]) || !$_GET ["ranking"]) echo ' class="active"';?>><a
						href="./">Home</a></li>
					<li
						<?php if (isset ( $_GET ["ranking"] ) && $_GET ["ranking"]) echo ' class="active"';?>><a
						href="./index.php?ranking=1">ランキング</a></li>
					<li><a href="http://atcoder.jp/">AtCoder (公式)</a></li>
					<li><a href="http://ichyo.jp/aoj-icpc/">AOJ-ICPC</a></li>
					<li><a href="http://twitter.com/kenkoooo">お問い合わせ</a></li>
				</ul>
			</div>
			<!--/.nav-collapse -->
		</div>
	</nav>
	<div class="container">

		<div class="page-header">
			<h1>AtCoder Problems</h1>
			<p class="lead">
				<a href="http://atcoder.jp/">AtCoder</a>の非公式過去問集です。User
				IDからACした問題を検索できます。(更新処理は5分ごと)
			</p>
			<p>
				<a href="https://twitter.com/share" class="twitter-share-button"
					data-url="http://kenkoooo.com/atcoder/">Tweet</a>
				<script>!function(d,s,id){var js,fjs=d.getElementsByTagName(s)[0],p=/^http:/.test(d.location)?'http':'https';if(!d.getElementById(id)){js=d.createElement(s);js.id=id;js.src=p+'://platform.twitter.com/widgets.js';fjs.parentNode.insertBefore(js,fjs);}}(document, 'script', 'twitter-wjs');</script>
			</p>
			<form action="index.php" method="get" class="form-inline"
				<?php
				if ((isset ( $_GET ["ranking"] ) && $_GET ["ranking"]) || (isset ( $_GET ["short_fast"] ) && $_GET ["short_fast"])) {
					echo ' style="visibility: hidden; height: 0;"';
				}
				
				?>>
				<div class="form-group">
					<label>User ID:</label>
					<?php echo '<input type="text" class="form-control" id="exampleInputEmail1" placeholder="User ID" name="name" value="' . $user_name . '">'; ?>
				</div>
				<div class="form-group">
					<label>ライバル:</label>
					<?php echo '<input type="text" class="form-control" id="exampleInputEmail1" placeholder="User ID" name="rivals" value="' . $rivals . '">'; ?>
				</div>
				<div class="form-group">
					<?php
					echo '<input type="radio" name="category" value="1" ';
					if (isset ( $_GET ["category"] ) && $_GET ["category"]) {
						echo 'checked';
					}
					echo '/>';
					echo 'カテゴリ';
					echo '<input type="radio" name="category" value="0" ';
					if (! isset ( $_GET ["category"] ) || ! $_GET ["category"]) {
						echo 'checked';
					}
					echo '/>';
					echo 'リスト';
					?>
				</div>
				<button type="submit" class="btn btn-default">Search</button>
			</form>

		</div>
	</div>

	<!-- div class="container" style="visibility: hidden; height: 0;"-->
		<?php
		
		if (isset ( $_GET ["ranking"] ) && $_GET ["ranking"]) {
			listRanking ( $ranking );
		} else if (isset ( $_GET ["short_fast"] ) && $_GET ["short_fast"]) {
			listShortFast ( $short_fast, $_GET ["short_fast"] );
		} else if (isset ( $_GET ["category"] ) && $_GET ["category"]) {
			listABC ( $problemArray, '/abc[0-9]*/i' );
			listABC ( $problemArray, '/arc[0-9]*/i' );
			listOther ( $problemArray );
		} else {
			listMode ( $problemArray );
		}
		?>

<?php if($user_name==="debug"){echo "<pre>";var_dump($problemArray);echo "</pre>";}?>

	<!-- /container -->


	<!-- Bootstrap core JavaScript
    ================================================== -->
	<!-- Placed at the end of the document so the pages load faster -->
	<script
		src="https://ajax.googleapis.com/ajax/libs/jquery/1.11.2/jquery.min.js"></script>
	<script src="js/bootstrap.min.js"></script>
	<script src="js/dataTables.bootstrap.js"></script>
	<script src="js/jquery.dataTables.min.js"></script>
	<script type="text/javascript">

	$(document).ready(function() {
	    $('#list').dataTable({
	        "bPaginate": false,
	        "bLengthChange": false,
	        "bFilter": true,
	        "bInfo": false,
	        "bAutoWidth": false,
			"aaSorting": [[ 3, "asc" ]]
	      });
	    $('#ranking').dataTable({
	        "bPaginate": false,
	        "bLengthChange": false,
	        "bFilter": true,
	        "bInfo": false,
	        "bAutoWidth": false,
			"aaSorting": [[ 0, "asc" ]]
	      });
	      $('#short').dataTable({
		        "bPaginate": false,
		        "bLengthChange": false,
		        "bFilter": true,
		        "bInfo": false,
		        "bAutoWidth": false,
				"aaSorting": [[ 2, "asc" ]]
		      });
	} );
	
	</script>
</body>
</html>